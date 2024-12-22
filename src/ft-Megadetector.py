import os
import json
import torch
from PIL import Image
from torchvision import datasets, transforms
from PytorchWildlife.models import classification as pw_classification
from tqdm import tqdm
from torch.utils.data import random_split
from src.dataset import NACTIAnnotationDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# We use scikit-learn metrics for precision, recall, and f1-score.
from sklearn.metrics import precision_score, recall_score, f1_score

################################################################################
# Data transformations
################################################################################
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

################################################################################
# Collate function to remove invalid samples
################################################################################
def collate_fn_remove_none(batch):
    """
    Custom collate function to remove None samples.
    batch is a list of (image, target) tuples.
    If image or target is None, remove that sample.
    Return None, None if all items are None.
    """
    filtered_batch = [item for item in batch if item is not None]
    if len(filtered_batch) == 0:
        return None, None
    imgs, tgts = zip(*filtered_batch)
    return list(imgs), list(tgts)

################################################################################
# Train / Validate / Test functions
################################################################################
def train_one_epoch(model,
                    loader,
                    optimizer,
                    criterion,
                    device,
                    writer,
                    epoch,
                    global_step_start=0,
                    log_interval=5):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model to train.
        loader (DataLoader): DataLoader for the training set.
        optimizer (Optimizer): The optimizer to update model parameters.
        criterion (nn.Module): The loss function.
        device (torch.device): Device on which to perform training.
        writer (SummaryWriter): TensorBoard writer for logging (can be None if not used).
        epoch (int): Current epoch index (for logging).
        global_step_start (int): The global step counter at the beginning of this epoch.
        log_interval (int): Batch interval for logging.

    Returns:
        (float, float, int) : A tuple of (epoch_loss, epoch_acc, new_global_step).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    global_step = global_step_start

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch} [Train]")):
        if batch[0] is None:
            continue

        global_step += 1
        images, targets = batch

        images = torch.stack(images).to(device)
        # get the labels from the target
        # the target is a dictionary with keys: boxes, labels, common_name
        labels = [t["labels"][0] for t in targets]
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        print(f"labels: {labels}")

        outputs = model(images)
        print(f"outputs: {outputs}")

        loss = criterion(outputs, labels)
        print(f"loss: {loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate epoch loss
        running_loss += loss.item() * images.size(0)

        # Calculate batch accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Optional: log batch metrics
        if writer is not None and ((batch_idx + 1) % log_interval == 0):
            batch_acc = (predicted == labels).float().mean().item()
            writer.add_scalar('Train/Loss_batch', loss.item(), global_step)
            writer.add_scalar('Train/Accuracy_batch', batch_acc, global_step)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0

    return epoch_loss, epoch_acc, global_step


def validate(model, loader, criterion, device, writer, epoch):
    """
    Validates the model over the validation set.

    Args:
        model (nn.Module): The PyTorch model to validate.
        loader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): The loss function for computing validation loss.
        device (torch.device): Device on which to perform validation.
        writer (SummaryWriter): TensorBoard writer for logging (can be None if not used).
        epoch (int): Current epoch index (for logging).

    Returns:
        tuple: (val_loss, val_acc, val_precision, val_recall, val_f1)
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Epoch {epoch} [Val]"):
            if batch[0] is None:
                continue
            images, targets = batch

            images = torch.stack(images).to(device)
            labels = [t["labels"][0] for t in targets]
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    val_loss /= val_total if val_total > 0 else 1
    val_acc = val_correct / val_total if val_total > 0 else 0

    # Compute macro precision/recall/F1
    val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Log validation metrics if writer is not None
    if writer is not None:
        writer.add_scalar('Val/Loss_epoch', val_loss, epoch)
        writer.add_scalar('Val/Accuracy_epoch', val_acc, epoch)
        writer.add_scalar('Val/Precision_epoch', val_precision, epoch)
        writer.add_scalar('Val/Recall_epoch', val_recall, epoch)
        writer.add_scalar('Val/F1_epoch', val_f1, epoch)

    return val_loss, val_acc, val_precision, val_recall, val_f1


def test_model(model, loader, criterion, device):
    """
    Tests the model over a test set and returns detailed metrics.

    Args:
        model (nn.Module): The PyTorch model to test.
        loader (DataLoader): DataLoader for the test set.
        criterion (nn.Module): The loss function for computing test loss.
        device (torch.device): Device on which to perform testing.

    Returns:
        dict: {
            'loss': float,
            'acc': float,
            'precision': float,
            'recall': float,
            'f1': float
        }
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    all_test_preds = []
    all_test_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            if batch[0] is None:
                continue

            images, targets = batch
            images = torch.stack(images).to(device)
            labels = [t["labels"][0] for t in targets]
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

            all_test_preds.extend(predicted.cpu().tolist())
            all_test_labels.extend(labels.cpu().tolist())

    # Compute final metrics
    test_loss /= test_total if test_total > 0 else 1
    test_acc = test_correct / test_total if test_total > 0 else 0

    test_precision = precision_score(all_test_labels, all_test_preds, average='macro', zero_division=0)
    test_recall = recall_score(all_test_labels, all_test_preds, average='macro', zero_division=0)
    test_f1 = f1_score(all_test_labels, all_test_preds, average='macro', zero_division=0)

    metrics = {
        'loss': test_loss,
        'acc': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
    }
    return metrics

################################################################################
# Main script
################################################################################
if __name__ == "__main__":
    # Choose your device and create the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = pw_classification.AI4GAmazonRainforest(device=device)
    # This model is set for 36 classes by default
    num_classes = 36

    # Prepare dataset
    dataset = NACTIAnnotationDataset(
        image_dir=r"F:\DATASET\NACTI\images\nacti_part0",
        json_path=r"E:\result\json\detection\part0output.json",
        csv_path=r"F:\DATASET\NACTI\meta\nacti_metadata_part0.csv",
        transforms=transform
    )

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn_remove_none)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn_remove_none)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn_remove_none)

    # Define optimizer & loss
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    # Set up TensorBoard
    writer = SummaryWriter()
    num_epochs = 10

    # Global step
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        #----------------------#
        # 1. Train one epoch
        #----------------------#
        train_epoch_loss, train_epoch_acc, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, writer, epoch,
            global_step_start=global_step
        )
        print(f"[Train] Epoch {epoch}/{num_epochs} | Loss: {train_epoch_loss:.4f} | Acc: {train_epoch_acc:.4f}")

        # log epoch-level metrics here
        writer.add_scalar('Train/Loss_epoch', train_epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy_epoch', train_epoch_acc, epoch)

        #----------------------#
        # 2. Validate
        #----------------------#
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate(
            model, val_loader, criterion, device, writer, epoch
        )
        print(f"[Val]   Epoch {epoch}/{num_epochs} | "
              f"Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | "
              f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")

    #----------------------#
    # Test after all epochs
    #----------------------#
    test_metrics = test_model(model, test_loader, criterion, device)
    print(f"[Test]  Loss: {test_metrics['loss']:.4f} | Acc: {test_metrics['acc']:.4f} | "
          f"Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f}")

    # You can also log test metrics to TensorBoard if desired:
    writer.add_scalar('Test/Loss', test_metrics['loss'], 0)
    writer.add_scalar('Test/Accuracy', test_metrics['acc'], 0)
    writer.add_scalar('Test/Precision', test_metrics['precision'], 0)
    writer.add_scalar('Test/Recall', test_metrics['recall'], 0)
    writer.add_scalar('Test/F1', test_metrics['f1'], 0)

    writer.close()

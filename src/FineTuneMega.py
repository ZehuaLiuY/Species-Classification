import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from PytorchWildlife.models import classification as pw_classification
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from dataset import NACTIAnnotationDataset
import numpy as np
from lossFunction import FocalLoss

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def collate_fn_remove_none(batch):
    """
    Custom collate function that removes None samples.

    Args:
        batch (list): A list of (image, target) tuples, possibly containing None.

    Returns:
        (list or None, list or None):
            - If valid samples exist, returns (list_of_images, list_of_targets).
            - If all are None, returns (None, None).
    """
    filtered_batch = [item for item in batch if item is not None]
    if len(filtered_batch) == 0:
        return None, None
    imgs, tgts = zip(*filtered_batch)
    return list(imgs), list(tgts)

def pil_collect_fn(batch):
    """
    Custom collate function that returns a list of PIL.Image and target_dict.

    Args:
        batch (list): A list of (PIL.Image, target_dict) tuples.

    Returns:
        (list, list): List of PIL.Image and list of target_dict.
    """
    imgs, tgts = zip(*batch)
    return list(imgs), list(tgts)


def train_one_epoch(model,
                    loader,
                    optimizer,
                    criterion,
                    device,
                    writer,
                    epoch,
                    global_step_start=0,
                    log_interval=5,
                    transform=None
):
    """
    Train for one epoch in a bounding-box-based workflow.

    Args:
        model (nn.Module): The classification model to be trained.
        loader (DataLoader): Dataloader returning a batch of (PIL.Image, target_dict).
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (nn.Module): Loss function, e.g., CrossEntropyLoss.
        device (torch.device): CPU or GPU device.
        writer (SummaryWriter): TensorBoard writer for logging (optional).
        epoch (int): Current epoch number (for logging).
        global_step_start (int): The global step counter at the start of this epoch.
        log_interval (int): How often (in batches) to log metrics to TensorBoard.
        transform (callable): A torchvision transform (Resize, Normalize, etc.)
            applied to each cropped bounding box.

    Returns:
        (float, float, int):
            - epoch_loss: Average loss over all bounding boxes in this epoch.
            - epoch_acc: Accuracy over all bounding boxes in this epoch.
            - global_step: Updated global step after this epoch.
    """
    model.train()
    running_loss = 0.0
    running_samples = 0
    correct = 0
    total = 0
    global_step = global_step_start

    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc=f"Epoch {epoch} [Train]")):
        if images is None or targets is None:
            continue

        all_crops = []
        all_labels = []

        for i in range(len(images)):
            # 'images[i]' is a PIL.Image if the dataset does not apply transforms
            pil_img = images[i]
            target_dict = targets[i]

            boxes = target_dict["boxes"]    # shape: [N, 4] => [x, y, w, h]
            labels = target_dict["labels"]  # shape: [N]

            if boxes.size(0) == 0:
                continue

            # Process each bounding box
            for j in range(boxes.size(0)):
                x1, y1, w, h = boxes[j]
                x2 = x1 + w
                y2 = y1 + h

                x1_, y1_, x2_, y2_ = map(int, [x1, y1, x2, y2])
                # Skip invalid or out-of-bounds boxes
                if x1_ < 0 or y1_ < 0 or x2_ <= x1_ or y2_ <= y1_:
                    continue

                # 1) Crop the bounding-box region
                cropped_pil = pil_img.crop((x1_, y1_, x2_, y2_))

                # 2) Apply the global transform (e.g. Resize, Normalize)
                if transform:
                    cropped_tensor = transform(cropped_pil).to(device)
                else:
                    # Minimal fallback: just convert to tensor
                    cropped_tensor = transforms.ToTensor()(cropped_pil).to(device)

                all_crops.append(cropped_tensor)
                all_labels.append(labels[j].item())

        if len(all_crops) == 0:
            continue

        # Stack into one big batch: [M, C, H, W]
        batch_crops = torch.stack(all_crops, dim=0)
        batch_labels = torch.tensor(all_labels, dtype=torch.long, device=device)

        # Forward & backward
        optimizer.zero_grad()
        outputs = model(batch_crops)  # [M, num_classes]
        loss = criterion(outputs, batch_labels)

        batch_loss = loss.item()
        batch_num = batch_labels.size(0)
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        running_loss += batch_loss
        running_samples += batch_num

        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == batch_labels).sum().item()
        total += batch_labels.size(0)

        # Log batch metrics
        if writer is not None and ((batch_idx + 1) % log_interval == 0):
            batch_acc = (predicted == batch_labels).float().mean().item()
            writer.add_scalar('Train/Loss_batch', batch_loss, global_step)
            writer.add_scalar('Train/Accuracy_batch', batch_acc, global_step)

        global_step += 1

    # Compute epoch metrics
    epoch_loss = running_loss / max(running_samples, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc, global_step


def validate(model, loader, criterion, device, writer, epoch, transform=None):
    """
    Validation step for batch_size=1, bounding-box-level classification.
    Includes mAP calculation for multi-label classification.

    Args:
        model (nn.Module): The classification model to validate.
        loader (DataLoader): Dataloader returning a batch of (PIL.Image, target_dict).
        criterion (nn.Module): Loss function.
        device (torch.device): CPU or GPU device.
        writer (SummaryWriter): TensorBoard writer for logging (optional).
        epoch (int): Current epoch number (for logging).
        transform (callable): The global transform for bounding box crops.

    Returns:
        dict: {
            'loss': float,
            'acc': float,
            'precision': float,
            'recall': float,
            'f1': float,
            'mAP': float
        }
    """
    model.eval()
    val_running_loss = 0.0
    val_running_total = 0
    val_correct = 0
    val_total = 0

    all_val_preds = []
    all_val_labels = []
    all_val_probs = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(loader, desc=f"Epoch {epoch} [Val]")):
            if images is None or targets is None:
                continue

            all_crops = []
            all_labels = []

            for i in range(len(images)):
                pil_img = images[i]
                target_dict = targets[i]
                boxes = target_dict["boxes"]
                labels = target_dict["labels"]

                if boxes.size(0) == 0:
                    continue

                for j in range(boxes.size(0)):
                    x1, y1, w, h = boxes[j]
                    x2 = x1 + w
                    y2 = y1 + h

                    x1_, y1_, x2_, y2_ = map(int, [x1, y1, x2, y2])
                    if x1_ < 0 or y1_ < 0 or x2_ <= x1_ or y2_ <= y1_:
                        continue

                    cropped_pil = pil_img.crop((x1_, y1_, x2_, y2_))
                    if transform:
                        cropped_tensor = transform(cropped_pil).to(device)
                    else:
                        cropped_tensor = transforms.ToTensor()(cropped_pil).to(device)

                    all_crops.append(cropped_tensor)
                    all_labels.append(labels[j].item())

            if len(all_crops) == 0:
                continue

            batch_crops = torch.stack(all_crops, dim=0)
            batch_labels = torch.tensor(all_labels, dtype=torch.long, device=device)

            outputs = model(batch_crops)
            loss = criterion(outputs, batch_labels)

            batch_loss = loss.item()
            batch_num = batch_labels.size(0)

            val_running_loss += batch_loss
            val_running_total += batch_num

            _, predicted = torch.max(outputs, dim=1)
            val_correct += (predicted == batch_labels).sum().item()
            val_total += batch_labels.size(0)

            all_val_preds.extend(predicted.cpu().tolist())
            all_val_labels.extend(batch_labels.cpu().tolist())
            all_val_probs.extend(outputs.softmax(dim=1).cpu().tolist())

    val_loss = val_running_loss / max(val_running_total, 1)
    val_acc = val_correct / max(val_total, 1)

    # Calculate precision, recall, F1-score
    val_precision = precision_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
    val_recall = recall_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
    val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)

    # Calculate mAP
    all_val_labels_one_hot = torch.nn.functional.one_hot(
        torch.tensor(all_val_labels), num_classes=49
    ).cpu().numpy()
    all_val_probs_np = torch.tensor(all_val_probs).cpu().numpy()

    # appeared class
    appeared_class = np.where(all_val_labels_one_hot.sum(axis=0) > 0)[0]
    if len(appeared_class) == 0:
        print("No appeared class in validation.")


    mAP = 0
    try:
        filtered_labels = all_val_labels_one_hot[:, appeared_class]
        filtered_probs = all_val_probs_np[:, appeared_class]
        mAP = average_precision_score(filtered_labels, filtered_probs, average="weighted")
    except ValueError:
        print("Error calculating mAP. Ensure non-empty predictions.")

    if writer is not None:
        writer.add_scalar('Val/Loss_epoch', val_loss, epoch)
        writer.add_scalar('Val/Accuracy_epoch', val_acc, epoch)
        writer.add_scalar('Val/Precision_epoch', val_precision, epoch)
        writer.add_scalar('Val/Recall_epoch', val_recall, epoch)
        writer.add_scalar('Val/F1_epoch', val_f1, epoch)
        writer.add_scalar('Val/mAP_epoch', mAP, epoch)

    metrics = {
        'loss': val_loss,
        'acc': val_acc,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1,
        'mAP': mAP,
    }
    return metrics


if __name__ == "__main__":
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model (this is a classification model from PytorchWildlife)
    model = pw_classification.AI4GAmazonRainforest(version="v1", device=device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # change the models number of classes to 46
    # model.num_cls = 46
    num_features = model.net.classifier.in_features
    # print(f"Number of features in the model: {num_features}") 2048
    model.net.classifier = torch.nn.Linear(num_features, 49)
    # model.net.classifier = torch.nn.Sequential(
    #     torch.nn.Linear(num_features, 512),
    #     torch.nn.ReLU(),
    #     torch.nn.Dropout(0.5),
    #     torch.nn.Linear(512, 46)
    # )
    print("Initialized model")

    print("Loading dataset...")
    dataset = NACTIAnnotationDataset(
        image_dir=r"F:\DATASET\NACTI\images",
        json_path=r"E:\result\json\detection\detection_filtered.json",
        csv_path=r"F:/DATASET/NACTI/meta/nacti_metadata_balanced.csv"
    )

    # Split dataset into train, val, test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Set DataLoader with batch_size=1
    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              shuffle=True,
                              collate_fn=pil_collect_fn)
    val_loader = DataLoader(val_dataset,
                            batch_size=8,
                            shuffle=False,
                            collate_fn=pil_collect_fn)

    print("DataLoaders created.")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = CrossEntropyLoss()
    writer = SummaryWriter()
    num_epochs = 1000
    global_step = 0
    patience = 10
    no_improvements = 0
    delta = 0.005

    # best f1 for saving the model
    best_f1 = 0

    for epoch in range(1, num_epochs + 1):
        # Train
        train_epoch_loss, train_epoch_acc, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, writer, epoch,
            global_step_start=global_step,
            transform=transform
        )
        print(f"[Train] Epoch {epoch}/{num_epochs} | "
              f"Loss: {train_epoch_loss:.4f} | Acc: {train_epoch_acc:.4f}")

        writer.add_scalar('Train/Loss_epoch', train_epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy_epoch', train_epoch_acc, epoch)

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, writer, epoch, transform=transform
        )
        print(f"[Validation]  Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['acc']:.4f} | "
              f"Precision: {val_metrics['precision']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | "
              f"mAP: {val_metrics['mAP']:.4f}")

        # save the best model
        if val_metrics['f1'] > best_f1 + delta:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"Best model saved with F1: {best_f1:.4f}, in Epoch: {epoch}")
            no_improvements = 0
        else:
            no_improvements += 1
            if no_improvements >= patience:
                print(f"Early stopping at Epoch: {epoch}")
                break

    # save final the model
    torch.save(model.state_dict(), "models/final_model.pth")
    print("Model saved.")

import torch
from torchvision import transforms
from PytorchWildlife.models import classification as pw_classification
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
from dataset import NACTIAnnotationDataset


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
    correct = 0
    total = 0
    global_step = global_step_start

    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc=f"Epoch {epoch} [Train]")):
        if images is None or targets is None:
            continue

        all_crops = []
        all_labels = []

        for i in range(len(images)):
            if images[i] is None or targets[i] is None:
                continue

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
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == batch_labels).sum().item()
        total += batch_labels.size(0)

        # Log batch metrics
        if writer is not None and ((batch_idx + 1) % log_interval == 0):
            batch_loss = loss.item()
            batch_acc = (predicted == batch_labels).float().mean().item()
            writer.add_scalar('Train/Loss_batch', batch_loss, global_step)
            writer.add_scalar('Train/Accuracy_batch', batch_acc, global_step)

        global_step += 1

    # Compute epoch metrics
    epoch_loss = running_loss / max(len(loader), 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc, global_step


def validate(model, loader, criterion, device, writer, epoch, transform=None):
    """
    Validation step for batch_size=1, bounding-box-level classification.
    No backpropagation, just forward passes to evaluate loss and accuracy.

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
            'f1': float
        }
    """
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(loader, desc=f"Epoch {epoch} [Val]")):
            if images is None or targets is None:
                continue

            all_crops = []
            all_labels = []

            for i in range(len(images)):
                if images[i] is None or targets[i] is None:
                    continue

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

            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            val_correct += (predicted == batch_labels).sum().item()
            val_total += batch_labels.size(0)

            all_val_preds.extend(predicted.cpu().tolist())
            all_val_labels.extend(batch_labels.cpu().tolist())

    val_loss = val_running_loss / max(len(loader), 1)
    val_acc = val_correct / max(val_total, 1)

    val_precision = precision_score(all_val_labels, all_val_preds, average='macro', zero_division=0)
    val_recall = recall_score(all_val_labels, all_val_preds, average='macro', zero_division=0)
    val_f1 = f1_score(all_val_labels, all_val_preds, average='macro', zero_division=0)

    if writer is not None:
        writer.add_scalar('Val/Loss_epoch', val_loss, epoch)
        writer.add_scalar('Val/Accuracy_epoch', val_acc, epoch)
        writer.add_scalar('Val/Precision_epoch', val_precision, epoch)
        writer.add_scalar('Val/Recall_epoch', val_recall, epoch)
        writer.add_scalar('Val/F1_epoch', val_f1, epoch)

    metrics = {
        'loss': val_loss,
        'acc': val_acc,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1,
    }
    return metrics


def test_model(model, loader, criterion, device, transform=None):
    """
    Test the model in a bounding-box-based workflow.

    Args:
        model (nn.Module): Classification model to test.
        loader (DataLoader): Dataloader returning a batch of (PIL.Image, target_dict).
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        device (torch.device): CPU or GPU device.
        transform (callable): Global transform for bounding box crops (same as train/val).

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
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0

    all_test_preds = []
    all_test_labels = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="Testing")):
            if images is None or targets is None:
                continue

            all_crops = []
            all_labels = []

            for i in range(len(images)):
                if images[i] is None or targets[i] is None:
                    continue

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

                    x1_, y1_, x2_, y2_ = map(int, [x1, y1, x2, y2_])
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

            test_running_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            test_correct += (predicted == batch_labels).sum().item()
            test_total += batch_labels.size(0)

            all_test_preds.extend(predicted.cpu().tolist())
            all_test_labels.extend(batch_labels.cpu().tolist())

    test_loss = test_running_loss / max(len(loader), 1)
    test_acc = test_correct / max(test_total, 1)

    test_precision = precision_score(all_test_labels, all_test_preds, average='macro', zero_division=0)
    test_recall = recall_score(all_test_labels, all_test_preds, average='macro', zero_division=0)
    test_f1 = f1_score(all_test_labels, all_test_preds, average='macro', zero_division=0)

    return {
        'loss': test_loss,
        'acc': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
    }


if __name__ == "__main__":
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model (this is a classification model from PytorchWildlife)
    model = pw_classification.AI4GAmazonRainforest(device=device)

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
    model.net.classifier = torch.nn.Linear(num_features, 46)
    # model.net.classifier = torch.nn.Sequential(
    #     torch.nn.Linear(num_features, 512),
    #     torch.nn.ReLU(),
    #     torch.nn.Dropout(0.5),
    #     torch.nn.Linear(512, 46)
    # )

    dataset = NACTIAnnotationDataset(
        image_dir=r"F:\DATASET\NACTI\images\nacti_part0",
        json_path=r"E:\result\json\detection\part0output.json",
        csv_path=r"F:\DATASET\NACTI\meta\nacti_metadata_part0.csv",
        # transforms=transform  # Resizing each image to 512x512
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
                              collate_fn=collate_fn_remove_none)
    val_loader = DataLoader(val_dataset,
                            batch_size=8,
                            shuffle=False,
                            collate_fn=collate_fn_remove_none)
    test_loader = DataLoader(test_dataset,
                             batch_size=8,
                             shuffle=False,
                             collate_fn=collate_fn_remove_none)

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter()
    num_epochs = 5
    global_step = 0

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
              f"F1: {val_metrics['f1']:.4f}")

    # Test
    test_metrics = test_model(model, test_loader, criterion, device, transform=transform)
    print(f"[Test]  Loss: {test_metrics['loss']:.4f} | "
          f"Acc: {test_metrics['acc']:.4f} | "
          f"Precision: {test_metrics['precision']:.4f} | "
          f"Recall: {test_metrics['recall']:.4f} | "
          f"F1: {test_metrics['f1']:.4f}")

    # log test metrics to TensorBoard
    writer.add_scalar('Test/Loss', test_metrics['loss'], 0)
    writer.add_scalar('Test/Accuracy', test_metrics['acc'], 0)
    writer.add_scalar('Test/Precision', test_metrics['precision'], 0)
    writer.add_scalar('Test/Recall', test_metrics['recall'], 0)
    writer.add_scalar('Test/F1', test_metrics['f1'], 0)

    writer.close()

    # save the model
    torch.save(model.state_dict(), "fine_tuned_model_46_classes.pth")
    print("Model saved.")

import torch
from torchvision import transforms
from PytorchWildlife.models import classification as pw_classification
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
from dataset import NACTIAnnotationDataset


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


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
                    log_interval=5):
    """
    Train the model for one epoch in a batch_size=1 setting.
    For each image, we iterate through all bounding boxes, crop the region,
    feed it to the model, accumulate the loss, and then do a backward + optimizer step.

    Args:
        model (nn.Module): The classification model.
        loader (DataLoader): DataLoader (batch_size=1) for training.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        device (torch.device): GPU or CPU device to use.
        writer (SummaryWriter): For TensorBoard logging (optional).
        epoch (int): Current epoch index (for logging).
        global_step_start (int): Global step counter at the beginning of this epoch.
        log_interval (int): Interval (in batches) for logging to TensorBoard.

    Returns:
        (float, float, int): (epoch_loss, epoch_acc, new_global_step)
            - epoch_loss: Average loss across the entire epoch.
            - epoch_acc: Accuracy across all bounding boxes in the epoch.
            - new_global_step: The updated global step after this epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    global_step = global_step_start
    w_thres, h_thres = 2, 2
    # Since batch_size=1, loader returns 1 (image, target) per iteration
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch} [Train]")):
        if batch[0] is None:
            continue

        images, targets = batch
        if not images or not targets or images[0] is None or targets[0] is None:
            continue

        # We have a single image and its corresponding target
        img_tensor = images[0].to(device)         # shape: [C, H, W]
        target_dict = targets[0]

        boxes = target_dict["boxes"].to(device)   # shape: [N, 4] => (x1, y1, w, h)
        labels = target_dict["labels"].to(device) # shape: [N]

        if boxes.size(0) == 0:
            # No bounding boxes
            continue

        optimizer.zero_grad()

        sample_loss = 0.0
        sample_correct = 0
        sample_total = 0

        # Loop over each bounding box in this image
        for i in range(boxes.size(0)):
            x1, y1, w, h = boxes[i]
            x2 = x1 + w
            y2 = y1 + h

            # Convert to int for slicing
            x1_, y1_, x2_, y2_ = map(int, [x1, y1, x2, y2])

            # Simple boundary check
            if x1_ < 0 or y1_ < 0 or x2_ > img_tensor.shape[2] or y2_ > img_tensor.shape[1]:
                continue

            # Check for very small boxes
            if (x2_ - x1_) < w_thres or (y2_ - y1_) < h_thres:
                continue

            # Crop the region from img_tensor
            cropped_tensor = img_tensor[:, y1_:y2_, x1_:x2_]

            cropped_tensor = cropped_tensor.unsqueeze(0)
            single_label = labels[i].unsqueeze(0)

            # Forward pass
            outputs = model(cropped_tensor)
            loss = criterion(outputs, single_label)
            sample_loss += loss

            # Accuracy
            _, predicted = torch.max(outputs, 1)
            sample_correct += (predicted == single_label).sum().item()
            sample_total += 1

        if sample_total > 0:
            # Average loss across the bounding boxes in this single image
            sample_loss = sample_loss / sample_total

            sample_loss.backward()
            optimizer.step()

            running_loss += sample_loss.item()
            correct += sample_correct
            total += sample_total

            # Logging
            if writer is not None and ((batch_idx + 1) % log_interval == 0):
                batch_acc = sample_correct / sample_total
                writer.add_scalar('Train/Loss_batch', sample_loss.item(), global_step)
                writer.add_scalar('Train/Accuracy_batch', batch_acc, global_step)

        global_step += 1

    # Compute epoch-level metrics
    # epoch_loss can be averaged per image (len(loader)) or per box (total)
    epoch_loss = running_loss / len(loader) if len(loader) > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0

    return epoch_loss, epoch_acc, global_step


def validate(model, loader, criterion, device, writer, epoch):
    """
    Validation step for batch_size=1, bounding-box-level classification.
    No backpropagation, just forward passes to evaluate loss and accuracy.

    Args:
        model (nn.Module): The classification model.
        loader (DataLoader): DataLoader (batch_size=1) for validation.
        criterion (nn.Module): Loss function.
        device (torch.device): CPU or GPU device.
        writer (SummaryWriter): For TensorBoard logging (optional).
        epoch (int): Current epoch index (for logging).

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
    w_thres, h_thres = 2, 2
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch} [Val]")):
            if batch[0] is None:
                continue

            images, targets = batch
            if not images or not targets or images[0] is None or targets[0] is None:
                continue

            img_tensor = images[0].to(device)
            target_dict = targets[0]
            boxes = target_dict["boxes"].to(device)
            labels = target_dict["labels"].to(device)

            if boxes.size(0) == 0:
                continue

            sample_loss = 0.0
            sample_correct = 0
            sample_total = 0

            sample_preds = []
            sample_lbls = []

            for i in range(boxes.size(0)):
                x1, y1, w, h = boxes[i]
                x2 = x1 + w
                y2 = y1 + h
                x1_, y1_, x2_, y2_ = map(int, [x1, y1, x2, y2])

                if x1_ < 0 or y1_ < 0 or x2_ > img_tensor.shape[2] or y2_ > img_tensor.shape[1]:
                    continue
                # Check for very small boxes
                if (x2_ - x1_) < w_thres or (y2_ - y1_) < h_thres:
                    continue

                cropped_tensor = img_tensor[:, y1_:y2_, x1_:x2_]
                cropped_tensor = cropped_tensor.unsqueeze(0)
                single_label = labels[i].unsqueeze(0)

                outputs = model(cropped_tensor)
                loss = criterion(outputs, single_label)

                sample_loss += loss

                _, predicted = torch.max(outputs, 1)
                sample_correct += (predicted == single_label).sum().item()
                sample_total += 1

                sample_preds.append(predicted.item())
                sample_lbls.append(single_label.item())

            if sample_total > 0:
                sample_loss = sample_loss / sample_total
                val_running_loss += sample_loss.item()
                val_correct += sample_correct
                val_total += sample_total

                all_val_preds.extend(sample_preds)
                all_val_labels.extend(sample_lbls)

    val_loss = val_running_loss / len(loader) if len(loader) > 0 else 0.0
    val_acc = val_correct / val_total if val_total > 0 else 0.0

    val_precision = (precision_score(all_val_labels, all_val_preds,
                                      average='macro', zero_division=0)
                      if val_total > 0 else 0)
    val_recall = (recall_score(all_val_labels, all_val_preds,
                                average='macro', zero_division=0)
                   if val_total > 0 else 0)
    val_f1 = (f1_score(all_val_labels, all_val_preds,
                        average='macro', zero_division=0)
               if val_total > 0 else 0)

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


def test_model(model, loader, criterion, device):
    """
    Test the model (batch_size=1, bounding-box-level classification).
    Similar to validation, but we'll store predictions/labels to compute precision, recall, and F1.

    Args:
        model (nn.Module): Classification model.
        loader (DataLoader): DataLoader (batch_size=1) for testing.
        criterion (nn.Module): Loss function.
        device (torch.device): CPU or GPU device.

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
    w_thres, h_thres = 2, 2
    all_test_preds = []
    all_test_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            if batch[0] is None:
                continue

            images, targets = batch
            if not images or not targets or images[0] is None or targets[0] is None:
                continue

            img_tensor = images[0].to(device)
            target_dict = targets[0]
            boxes = target_dict["boxes"].to(device)
            labels = target_dict["labels"].to(device)

            if boxes.size(0) == 0:
                continue

            sample_loss = 0.0
            sample_correct = 0
            sample_total = 0

            sample_preds = []
            sample_lbls = []

            for i in range(boxes.size(0)):
                x1, y1, w, h = boxes[i]
                x2 = x1 + w
                y2 = y1 + h
                x1_, y1_, x2_, y2_ = map(int, [x1, y1, x2, y2])

                if x1_ < 0 or y1_ < 0 or x2_ > img_tensor.shape[2] or y2_ > img_tensor.shape[1]:
                    continue

                # Check for very small boxes
                if (x2_ - x1_) < w_thres or (y2_ - y1_) < h_thres:
                    continue

                cropped_tensor = img_tensor[:, y1_:y2_, x1_:x2_]
                cropped_tensor = cropped_tensor.unsqueeze(0)
                single_label = labels[i].unsqueeze(0)

                outputs = model(cropped_tensor)
                loss = criterion(outputs, single_label)
                sample_loss += loss

                _, predicted = torch.max(outputs, 1)
                sample_correct += (predicted == single_label).sum().item()
                sample_total += 1

                sample_preds.append(predicted.item())
                sample_lbls.append(single_label.item())

            if sample_total > 0:
                sample_loss = sample_loss / sample_total
                test_running_loss += sample_loss.item()
                test_correct += sample_correct
                test_total += sample_total

                all_test_preds.extend(sample_preds)
                all_test_labels.extend(sample_lbls)

    # Final metrics
    test_loss = test_running_loss / len(loader) if len(loader) > 0 else 0.0
    test_acc = test_correct / test_total if test_total > 0 else 0.0

    test_precision = (precision_score(all_test_labels, all_test_preds,
                                      average='macro', zero_division=0)
                      if test_total > 0 else 0)
    test_recall = (recall_score(all_test_labels, all_test_preds,
                                average='macro', zero_division=0)
                   if test_total > 0 else 0)
    test_f1 = (f1_score(all_test_labels, all_test_preds,
                        average='macro', zero_division=0)
               if test_total > 0 else 0)

    metrics = {
        'loss': test_loss,
        'acc': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
    }
    return metrics


if __name__ == "__main__":
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model (this is a classification model from PytorchWildlife)
    model = pw_classification.AI4GAmazonRainforest(device=device)

    # change the models number of classes to 46
    # model.num_cls = 46
    num_features = model.net.classifier.in_features
    # print(f"Number of features in the model: {num_features}") 2048
    model.net.classifier = torch.nn.Linear(num_features, 46)

    dataset = NACTIAnnotationDataset(
        image_dir=r"F:\DATASET\NACTI\images\nacti_part0",
        json_path=r"E:\result\json\detection\part0output.json",
        csv_path=r"F:\DATASET\NACTI\meta\nacti_metadata_part0.csv",
        transforms=transform  # Resizing each image to 512x512
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
                              batch_size=1,
                              shuffle=True,
                              collate_fn=collate_fn_remove_none)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=collate_fn_remove_none)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
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
            global_step_start=global_step
        )
        print(f"[Train] Epoch {epoch}/{num_epochs} | "
              f"Loss: {train_epoch_loss:.4f} | Acc: {train_epoch_acc:.4f}")

        writer.add_scalar('Train/Loss_epoch', train_epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy_epoch', train_epoch_acc, epoch)

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, writer, epoch
        )
        print(f"[Test]  Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['acc']:.4f} | "
              f"Precision: {val_metrics['precision']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f} | "
              f"F1: {val_metrics['f1']:.4f}")

    # Test
    test_metrics = test_model(model, test_loader, criterion, device)
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

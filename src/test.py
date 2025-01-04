import torch
from torchvision import transforms
from PytorchWildlife.models import classification as pw_classification
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
from dataset import NACTIAnnotationDataset
import argparse

parser = argparse.ArgumentParser(
    description="Republish Predicting Eye Fixations",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--model_path",
    type=str,
    help="Path to the model.pth file",
)

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


    metrics = {
        'loss': test_loss,
        'acc': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
    }
    return metrics

def main(args):
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load the fine-tuned classification model
    model = pw_classification.AI4GAmazonRainforest(device=device)
    checkpoint = torch.load(args.model_path, weights_only=True, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

    model.load_state_dict(state_dict, strict=False)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

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

    test_loader = DataLoader(test_dataset,
                             batch_size=8,
                             shuffle=False,
                             collate_fn=collate_fn_remove_none)

    writer = SummaryWriter()
    criterion = torch.nn.CrossEntropyLoss()

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

if __name__ == "__main__":
    main(parser.parse_args())

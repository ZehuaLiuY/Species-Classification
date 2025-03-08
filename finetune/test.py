import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import transforms
from PytorchWildlife.models import classification as pw_classification
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support, confusion_matrix
from dataset import NACTIAnnotationDataset
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

Class_names = {
    0: 'american black bear', 1: 'american marten', 2: 'american red squirrel', 3: 'black-tailed jackrabbit',
    4: 'bobcat', 5: 'california ground squirrel', 6: 'california quail', 7: 'cougar', 8: 'coyote', 9: 'dark-eyed junco',
    10: 'domestic cow', 11: 'domestic dog', 12: 'donkey', 13: 'dusky grouse', 14: 'eastern gray squirrel',
    15: 'elk', 16: 'ermine', 17: 'european badger', 18: 'gray fox', 19: 'gray jay', 20: 'horse',
    21: 'house wren', 22: 'long-tailed weasel', 23: 'moose', 24: 'mule deer', 25: 'nine-banded armadillo', 26: 'north american porcupine',
    27: 'north american river otter', 28: 'raccoon', 29: 'red deer', 30: 'red fox', 31: 'snowshoe hare',
    32: "steller's jay", 33: 'striped skunk', 34: 'unidentified accipitrid', 35: 'unidentified bird',
    36: 'unidentified chipmunk', 37: 'unidentified corvus', 38: 'unidentified deer', 39: 'unidentified deer mouse',
    40: 'unidentified mouse', 41: 'unidentified pack rat', 42: 'unidentified pocket gopher', 43: 'unidentified rabbit',
    44: 'virginia opossum', 45: 'wild boar', 46: 'wild turkey', 47: 'yellow-bellied marmot'
}

parser = argparse.ArgumentParser(
    description="Test the fine tuned model on the NACTI dataset.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--model_path",
    type=str,
    help="Path to the model.pth file",
)

parser.add_argument(
    "--train_type",
    choices=['single', 'ddp'], default='ddp',
    help="Choose training type: 'single' for single GPU or 'ddp' for DistributedDataParallel"
)

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
            'f1': float,
            'per_class_precision': list,
            'per_class_recall': list,
            'per_class_f1': list,
            'per_class_accuracy': list,
            'true_positives': list,
            'class_bias': list,
            'class_prevalence': list,
            'classes_order': list,
            'confusion_matrix': list of lists
        }
    """
    model.eval()
    test_running_loss = 0.0
    test_running_total = 0
    test_correct = 0
    test_total = 0

    all_test_preds = []
    all_test_labels = []
    results = []

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
            batch_loss = loss.item()
            batch_num = batch_labels.size(0)

            test_running_loss += batch_loss
            test_running_total += batch_num
            _, predicted = torch.max(outputs, dim=1)

            predicted_class_names = [Class_names[p.item()] for p in predicted]
            gt_class_names = [Class_names[l.item()] for l in batch_labels]

            for pred_name, gt_name in zip(predicted_class_names, gt_class_names):
                results.append({
                    'predicted_class': pred_name,
                    'ground_truth_class': gt_name
                })

            test_correct += (predicted == batch_labels).sum().item()
            test_total += batch_labels.size(0)

            all_test_preds.extend(predicted.cpu().tolist())
            all_test_labels.extend(batch_labels.cpu().tolist())

    test_loss = test_running_loss / max(len(loader), 1)
    test_acc = test_correct / max(test_total, 1)

    # per-class metrics
    class_prevalence = np.zeros(48, dtype=int)
    class_bias = np.zeros(48, dtype=int)
    for label in all_test_labels:
        class_prevalence[label] += 1
    for pred in all_test_preds:
        class_bias[pred] += 1
    print(f"Class prevalence: {class_prevalence}")
    print(f"Class bias: {class_bias}")

    test_precision = precision_score(all_test_labels, all_test_preds, average='weighted', zero_division=0)
    test_recall = recall_score(all_test_labels, all_test_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted', zero_division=0)

    per_class_prec, per_class_rec, per_class_f1, support = precision_recall_fscore_support(
        all_test_labels, all_test_preds, labels=range(48), zero_division=0
    )

    # calculate per-class accuracy
    true_positives = np.zeros(48, dtype=int)
    all_test_labels_np = np.array(all_test_labels)
    all_test_preds_np = np.array(all_test_preds)
    for i in range(48):
        true_positives[i] = np.sum((all_test_labels_np == i) & (all_test_preds_np == i))

    per_class_accuracy = np.zeros(48, dtype=float)
    for i in range(48):
        if class_prevalence[i] > 0:
            per_class_accuracy[i] = true_positives[i] / class_prevalence[i]
        else:
            per_class_accuracy[i] = 0.0

    metrics = {
        'loss': test_loss,
        'acc': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'per_class_precision': per_class_prec.tolist(),
        'per_class_recall': per_class_rec.tolist(),
        'per_class_f1': per_class_f1.tolist(),
        'per_class_accuracy': per_class_accuracy.tolist(),
        'true_positives': true_positives.tolist(),
        'class_bias': class_bias.tolist(),
        'class_prevalence': class_prevalence.tolist(),
        'classes_order': list(range(48))
    }

    # calculate confusion matrix
    cm = confusion_matrix(all_test_labels, all_test_preds, labels=list(range(48)))
    metrics['confusion_matrix'] = cm.tolist()

    # save detailed results to JSON file
    with open("../test_result/forcal_loss.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return metrics

def main(args):
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and checkpoint
    model = pw_classification.AI4GAmazonRainforest(device=device)
    num_features = model.net.classifier.in_features
    model.net.classifier = torch.nn.Linear(num_features, 48)

    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)

    # if using DistributedDataParallel, remove the "module." prefix
    if args.train_type == 'ddp':
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    else:
        state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = NACTIAnnotationDataset(
        image_dir=r"F:\DATASET\NACTI\images",
        json_path=r"E:\result\json\detection\formatted_file.json",
        csv_path=r"F:/DATASET/NACTI/meta/nacti_metadata_balanced.csv"
    )

    print("Constructing test dataset...")

    g = torch.Generator().manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Split dataset into train, val, test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], g
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=pil_collect_fn
    )

    criterion = torch.nn.CrossEntropyLoss()

    # Test the model
    test_metrics = test_model(model, test_loader, criterion, device, transform=transform)
    print(f"[Test]  Loss: {test_metrics['loss']:.4f} | "
          f"Acc: {test_metrics['acc']:.4f} | "
          f"Precision: {test_metrics['precision']:.4f} | "
          f"Recall: {test_metrics['recall']:.4f} | "
          f"F1: {test_metrics['f1']:.4f}")

    # Save test results to a text file
    with open("../test_result/forcal_loss.txt", "w", encoding="utf-8") as f:
        f.write("==== Test Results ====\n")
        f.write(f"Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"Overall Accuracy: {test_metrics['acc']:.4f}\n")
        f.write(f"Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"F1: {test_metrics['f1']:.4f}\n")

        f.write("=== Detailed Per-Class Metrics ===\n")
        header = (
            f"{'Class':<35}"
            f"{'Precision':>10}"
            f"{'True Pos.':>10}"
            f"{'Class Bias':>12}"
            f"{'Recall':>10}"
            f"{'Prevalence':>14}"
            f"{'F1 Score':>10}\n"
        )
        f.write(header)
        f.write("-" * (35 + 10 + 10 + 12 + 10 + 14 + 10) + "\n")

        for i in range(48):
            class_name = f"Class {i} ({Class_names[i]})"
            line = (
                f"{class_name:<35}"
                f"{test_metrics['per_class_precision'][i]:>10.4f}"
                f"{test_metrics['true_positives'][i]:>10}"
                f"{test_metrics['class_bias'][i]:>12}"
                f"{test_metrics['per_class_recall'][i]:>10.4f}"
                f"{test_metrics['class_prevalence'][i]:>14}"
                f"{test_metrics['per_class_f1'][i]:>10.4f}\n"
            )
            f.write(line)

        # Add overall metrics
        plt.figure(figsize=(15, 8))
        plt.bar(range(48), test_metrics['class_prevalence'], tick_label=[Class_names[i] for i in range(48)])
        plt.xticks(rotation=90)
        plt.xlabel("Class")
        plt.ylabel("Number of Images")
        plt.title("Class Prevalence Histogram")
        plt.tight_layout()
        plt.savefig("../test_result/class_prevalence_histogram.png")
        plt.close()

    # Plot confusion matrix
    cm = np.array(test_metrics['confusion_matrix'])
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(48)
    plt.xticks(tick_marks, [Class_names[i] for i in range(48)], rotation=90, fontsize=8)
    plt.yticks(tick_marks, [Class_names[i] for i in range(48)], fontsize=8)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("../test_result/confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    main(parser.parse_args())

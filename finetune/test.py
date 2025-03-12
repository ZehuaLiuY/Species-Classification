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
import os
import matplotlib.colors as colors

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
    44: 'vehicle', 45: 'virginia opossum', 46: 'wild boar', 47: 'wild turkey', 48: 'yellow-bellied marmot'
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
    choices=['single', 'ddp'], default='single',
    help="Choose training type: 'single' for single GPU or 'ddp' for DistributedDataParallel"
)

parser.add_argument(
    "--num_classes",
    type=int,
    default=49,
    help="Number of classes in the model (48 if vehicle is excluded, 49 if included)",
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

def filter_vehicle(subset):
    """
    filter out the vehicle class from the subset
    """
    filtered_indices = []
    for idx in tqdm(subset.indices, desc="Filtering"):
        _, target = subset.dataset[idx]
        if target.get("common_name", "").lower() != "vehicle":
            filtered_indices.append(idx)
    return filtered_indices

def test_model(model, loader, criterion, device, transform=None, num_class=None, class_names=None, remap=None, filter_vehicle_in_test=False):
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
            batch_labels_list = []
            batch_results = []

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
                    # 不进行 remap（remap 为 None），直接保留原始标签
                    original_label = labels[j].item()
                    mapped_label = remap[original_label] if remap is not None else original_label
                    batch_labels_list.append(mapped_label)
                    batch_results.append({
                        'original_label': original_label
                    })
            if len(all_crops) == 0:
                continue
            batch_crops = torch.stack(all_crops, dim=0)
            batch_labels = torch.tensor(batch_labels_list, dtype=torch.long, device=device)
            outputs = model(batch_crops)
            loss = criterion(outputs, batch_labels)
            batch_loss = loss.item()
            batch_num = batch_labels.size(0)

            test_running_loss += batch_loss
            test_running_total += batch_num
            _, predicted = torch.max(outputs, dim=1)

            all_test_preds.extend(predicted.cpu().tolist())
            all_test_labels.extend(batch_labels.cpu().tolist())
            for idx in range(len(batch_results)):
                pred_label = predicted[idx].item()
                gt_label = batch_labels[idx].item()
                batch_results[idx]['predicted_label'] = pred_label
                batch_results[idx]['ground_truth_label'] = gt_label
                batch_results[idx]['predicted_class'] = class_names[pred_label]
                batch_results[idx]['ground_truth_class'] = class_names[gt_label]
            results.extend(batch_results)
            test_correct += (predicted == batch_labels).sum().item()
            test_total += batch_labels.size(0)

    # if enabled, filter out the vehicle class
    if filter_vehicle_in_test:
        vehicle_idx = 44
        mask = np.array(all_test_labels) != vehicle_idx
        all_test_labels = np.array(all_test_labels)[mask].tolist()
        all_test_preds = np.array(all_test_preds)[mask].tolist()
        results = [r for r in results if r['ground_truth_class'].lower() != "vehicle"]

    test_loss = test_running_loss / max(len(loader), 1)
    test_acc = test_correct / max(test_total, 1)

    # 对于 49 类模型，定义有效标签为所有非 vehicle 的原始标签
    if num_class == 49:
        valid_labels = [i for i in range(49) if i != 44]
    else:
        valid_labels = sorted([label for label, name in Class_names.items() if name.lower() != "vehicle"])

    # if remap is not None, then remap the valid labels
    num_valid = len(valid_labels)
    class_prevalence = np.zeros(num_valid, dtype=int)
    class_bias = np.zeros(num_valid, dtype=int)

    # assume all_test_labels and all_test_preds are remapped
    for label in all_test_labels:
        if label in valid_labels:
            idx = valid_labels.index(label)
            class_prevalence[idx] += 1
    for pred in all_test_preds:
        if pred in valid_labels:
            idx = valid_labels.index(pred)
            class_bias[idx] += 1

    print(f"Class prevalence (non-vehicle): {class_prevalence}")
    print(f"Class bias (non-vehicle): {class_bias}")

    test_precision = precision_score(all_test_labels, all_test_preds, labels=valid_labels, average='weighted', zero_division=0)
    test_recall = recall_score(all_test_labels, all_test_preds, labels=valid_labels, average='weighted', zero_division=0)
    test_f1 = f1_score(all_test_labels, all_test_preds, labels=valid_labels, average='weighted', zero_division=0)

    per_class_prec, per_class_rec, per_class_f1, support = precision_recall_fscore_support(
        all_test_labels, all_test_preds, labels=valid_labels, zero_division=0
    )

    true_positives = np.zeros(num_valid, dtype=int)
    all_test_labels_np = np.array(all_test_labels)
    all_test_preds_np = np.array(all_test_preds)
    for i, lab in enumerate(valid_labels):
        true_positives[i] = np.sum((all_test_labels_np == lab) & (all_test_preds_np == lab))

    per_class_accuracy = np.zeros(num_valid, dtype=float)
    for i in range(num_valid):
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
        'classes_order': list(range(num_valid))
    }
    cm = confusion_matrix(all_test_labels, all_test_preds, labels=valid_labels)
    metrics['confusion_matrix'] = cm.tolist()

    os.makedirs("../test_result/json/49", exist_ok=True)
    with open("../test_result/json/49/FL_AdamW.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return metrics

def main(args):
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_class = args.num_classes

    # Load model and checkpoint
    model = pw_classification.AI4GAmazonRainforest(device=device)
    num_features = model.net.classifier.in_features
    model.net.classifier = torch.nn.Linear(num_features, num_class)

    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
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

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], g)
    print("finished splitting")

    # if the number of classes is 48, we need to filter out the vehicle class, and remap the labels
    if num_class == 48:
        print("filtering test dataset, excluding vehicle class...")
        filtered_test_indices = filter_vehicle(test_dataset)
        print("finished filtering")
        test_dataset = Subset(test_dataset.dataset, filtered_test_indices)

        # new mapping
        valid_old_labels = sorted([label for label, name in Class_names.items() if name.lower() != "vehicle"])
        new_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_old_labels)}
        display_class_names = {new_mapping[old_label]: Class_names[old_label] for old_label in valid_old_labels}
    else:
        print("using full test dataset, but for safety we can filter in test_model if needed")
        new_mapping = None
        display_class_names = Class_names

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=pil_collect_fn)
    criterion = torch.nn.CrossEntropyLoss()

    test_metrics = test_model(
        model, test_loader, criterion, device,
        transform=transform, num_class=num_class, class_names=display_class_names, remap=new_mapping,
        filter_vehicle_in_test=(num_class != 48) # only filter vehicle if the number of classes is 48
    )
    print(f"[Test]  Loss: {test_metrics['loss']:.4f} | Acc: {test_metrics['acc']:.4f} | Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f}")

    os.makedirs("../test_result/txt/49", exist_ok=True)
    with open("../test_result/txt/49/FL_AdamW.txt", "w", encoding="utf-8") as f:
        f.write("==== Test Results ====\n")
        f.write(f"Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"Overall Accuracy: {test_metrics['acc']:.4f}\n")
        f.write(f"Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"F1: {test_metrics['f1']:.4f}\n")
        f.write("=== Detailed Per-Class Metrics ===\n")
        header = f"{'Class':<35}{'Precision':>10}{'True Pos.':>10}{'Class Bias':>12}{'Recall':>10}{'Prevalence':>14}{'F1 Score':>10}\n"
        f.write(header)
        f.write("-" * (35+10+10+12+10+14+10) + "\n")
        # 对于 49 类，使用有效标签：[i for i in range(49) if i != 44]
        valid_labels = [i for i in range(49) if i != 44]
        for i, label in enumerate(valid_labels):
            class_name = f"Class {i} ({display_class_names[label]})"
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

    cm = np.array(test_metrics['confusion_matrix'])
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, norm=colors.LogNorm())
    plt.title("Confusion Matrix (non-vehicle)")
    plt.colorbar()
    tick_marks = np.arange(len(valid_labels))
    tick_labels = [f"{label}-{display_class_names[label]}" for label in valid_labels]
    plt.xticks(tick_marks, tick_labels, rotation=90, fontsize=8)
    plt.yticks(tick_marks, tick_labels, fontsize=8)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("../test_result/cm/49/FL_AdamW.png")
    plt.close()

if __name__ == "__main__":
    main(parser.parse_args())

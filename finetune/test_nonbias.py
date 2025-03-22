import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_fscore_support
from dataset import ena24Dataset

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

def pil_collect_fn(batch):
    """
    将每个样本返回的 (cropped_imgs, target) 分别收集成列表。
    """
    crop_lists, targets = zip(*batch)
    return list(crop_lists), list(targets)

# -------------------------
# 3. 定义测试函数
# -------------------------
def test_model(model, loader, criterion, device, transform=None, num_class=None, class_names=None, remap=None, filter_vehicle_in_test=False):
    model.eval()
    test_running_loss = 0.0
    test_running_total = 0
    test_correct = 0

    all_test_preds = []
    all_test_labels = []
    results = []

    with torch.no_grad():
        for batch_crop_lists, targets in tqdm(loader, desc="Testing"):
            # batch_crop_lists 是一个列表，每个元素为该样本的 cropped_imgs（一个列表）
            all_crops = []
            batch_labels_list = []
            batch_results = []

            for i in range(len(batch_crop_lists)):
                crop_list = batch_crop_lists[i]  # 该样本中所有裁剪图像
                target_dict = targets[i]
                labels = target_dict["labels"]  # tensor形式的标签列表
                if len(crop_list) == 0:
                    continue
                for j in range(len(crop_list)):
                    # 如果返回的是 PIL.Image，则用 transform 或 ToTensor 转为 tensor
                    cropped_img = crop_list[j]
                    if isinstance(cropped_img, Image.Image):
                        if transform:
                            cropped_tensor = transform(cropped_img).to(device)
                        else:
                            cropped_tensor = transforms.ToTensor()(cropped_img).to(device)
                    else:
                        cropped_tensor = cropped_img.to(device)
                    all_crops.append(cropped_tensor)
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
            test_running_loss += loss.item()
            test_running_total += batch_labels.size(0)
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

    test_loss = test_running_loss / max(len(loader), 1)
    test_acc = test_correct / max(test_running_total, 1)

    # 如有需要，过滤掉 Vehicle 类（假设 Vehicle 在 JSON 中的 id 为 9）
    if filter_vehicle_in_test:
        vehicle_idx = remap[9] if remap is not None and 9 in remap else 9
        mask = np.array(all_test_labels) != vehicle_idx
        all_test_labels = np.array(all_test_labels)[mask].tolist()
        all_test_preds = np.array(all_test_preds)[mask].tolist()
        results = [r for r in results if r['ground_truth_class'].lower() != "vehicle"]

    cm = confusion_matrix(all_test_labels, all_test_preds)
    metrics = {
        'loss': test_loss,
        'acc': test_acc,
        'confusion_matrix': cm.tolist()
    }

    os.makedirs("../test_result/json/48", exist_ok=True)
    with open("../test_result/json/48/CE_Adam.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return metrics

# -------------------------
# 4. 主测试脚本
# -------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    num_class = args.num_classes

    # 加载模型；这里示例使用 PytorchWildlife 中的模型
    from PytorchWildlife.models import classification as pw_classification
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

    # 定义 transform，用于将裁剪后的 PIL.Image 转换为 tensor 并归一化
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 加载整个测试数据集
    dataset = ena24Dataset(
        image_dir=r"F:/DATASET/ENA24-Detection/images",
        json_path=r"F:/DATASET/ENA24-Detection/metadata/ena24.json",
        transforms=None  # __getitem__ 内未做 transform，此处后续在 test_model 使用 transform
    )
    print("Loaded dataset with {} samples.".format(len(dataset)))

    # 如果模型类别为48（即需要过滤 vehicle），则过滤掉含 vehicle 的样本，并构建 remap
    # 根据之前的对比，这里假设与模型对应的 JSON 中的类别 id 为：
    # [1, 4, 6, 10, 11, 13, 14, 15, 16, 20, 21, 22]
    if num_class == 48:
        print("Filtering out samples containing vehicle...")
        filtered_indices = filter_vehicle(dataset)
        dataset = Subset(dataset, filtered_indices)
        valid_old_labels = sorted([1, 4, 6, 10, 11, 13, 14, 15, 16, 20, 21, 22])
        new_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_old_labels)}
        # 这里 class_names 映射使用你预先定义的 Class_names（确保数字对应正确）
        display_class_names = {new_mapping[old_label]: Class_names[old_label] for old_label in valid_old_labels}
    else:
        new_mapping = None
        display_class_names = Class_names

    test_loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=pil_collect_fn)
    criterion = torch.nn.CrossEntropyLoss()

    test_metrics = test_model(
        model, test_loader, criterion, device,
        transform=transform, num_class=num_class, class_names=display_class_names, remap=new_mapping,
        filter_vehicle_in_test=(num_class != 48)
    )
    print(f"[Test] Loss: {test_metrics['loss']:.4f} | Acc: {test_metrics['acc']:.4f}")

    # 绘制混淆矩阵
    cm = np.array(test_metrics['confusion_matrix'])
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(display_class_names))
    tick_labels = [f"{i}-{display_class_names[i]}" for i in range(len(display_class_names))]
    plt.xticks(tick_marks, tick_labels, rotation=90, fontsize=8)
    plt.yticks(tick_marks, tick_labels, fontsize=8)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    os.makedirs("../test_result/cm/48", exist_ok=True)
    plt.savefig("../test_result/cm/48/CE_Adam.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the model on the entire ena24 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.pth file")
    parser.add_argument("--train_type", choices=['single', 'ddp'], default='single', help="Training type")
    parser.add_argument("--num_classes", type=int, default=48, help="Number of classes in the model")
    args = parser.parse_args()
    main(args)
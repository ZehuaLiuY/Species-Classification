import os
import shutil
import torch
from torch.utils.data import random_split
from src.dataset import NACTIAnnotationDataset

dataset = NACTIAnnotationDataset(
    image_dir=r"F:\DATASET\NACTI\images",
    json_path=r"E:\result\json\detection\formatted_file.json",
    csv_path=r"F:/DATASET/NACTI/meta/nacti_metadata_balanced.csv"
)

total_len = len(dataset)
train_size = int(0.8 * total_len)
val_size = int(0.1 * total_len)
test_size = total_len - train_size - val_size

g = torch.Generator()
g.manual_seed(0)
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size], generator=g)

def extract_images_for_class(subset, class_label, image_dir, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for idx in subset.indices:
        # get the sample
        sample = subset.dataset.samples[idx]
        target = sample["target"]

        if target["labels"].numel() > 0 and (target["labels"] == class_label).any().item():
            rel_path = sample["rel_path"]
            src_path = os.path.join(image_dir, rel_path)
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_folder)
            else:
                print(f"image not exist：{src_path}")

desired_label = 1

train_dest = r"E:\result\set\train_label_1"
val_dest = r"E:\result\set\val_label_1"
test_dest = r"E:\result\set\test_label_1"
# make sure the destination folders exist
os.makedirs(train_dest, exist_ok=True)
os.makedirs(val_dest, exist_ok=True)
os.makedirs(test_dest, exist_ok=True)

# get the images for the desired label
extract_images_for_class(train_dataset, desired_label, r"F:\DATASET\NACTI\images", train_dest)
extract_images_for_class(val_dataset, desired_label, r"F:\DATASET\NACTI\images", val_dest)
extract_images_for_class(test_dataset, desired_label, r"F:\DATASET\NACTI\images", test_dest)

print("Finish extracting images for class:", desired_label)

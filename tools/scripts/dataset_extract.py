import os
import shutil
from PIL import Image

from finetune.dataset import ENA24Dataset
from src.dataset import *

# Create the dataset instance with the specified paths.
dataset = NACTIAnnotationDataset(
    image_dir=r"F:\DATASET\NACTI\images",
    json_path=r"E:\result\json\detection\detection_filtered.json",
    csv_path=r"F:/DATASET/NACTI/meta/nacti_metadata_balanced.csv"
)
# dataset = ENA24Dataset(
#     image_dir=r"F:/DATASET/ENA24-Detection/images",
#     json_path=r"F:/DATASET/ENA24-Detection/metadata/ena24_updated.json"
# )

# Define output directories for full images for specific labels.
output_dir_label46 = r"E:\result\nacti_image\wildturkey"
output_dir_label20 = r"E:\result\nacti_image\horse"

# Define output directories for cropped images for each label.
crop_dir_label46 = r"E:\result\nacti_image\cropped\wildturkey_crop"
crop_dir_label20 = r"E:\result\nacti_image\cropped\horse_crop"

# Create all directories if they do not exist.
os.makedirs(output_dir_label46, exist_ok=True)
os.makedirs(output_dir_label20, exist_ok=True)
os.makedirs(crop_dir_label46, exist_ok=True)
os.makedirs(crop_dir_label20, exist_ok=True)

# Iterate over each sample in the dataset.
for sample in dataset.samples:
    rel_path = sample["rel_path"]
    target = sample["target"]

    label = target["labels"][0].item()
    src_path = os.path.join(dataset.image_dir, rel_path)
    img = Image.open(src_path)
    box = target["boxes"][0].tolist()
    x, y, w, h = box
    cropped_img = img.crop((x, y, x + w, y + h))

    if label == 46:
        dst_path = os.path.join(output_dir_label46, os.path.basename(src_path))
        shutil.copy(src_path, dst_path)
        print(f"Copied {src_path} to {dst_path}")

        crop_dst_path = os.path.join(crop_dir_label46, os.path.basename(src_path))
        cropped_img.save(crop_dst_path)
        print(f"Cropped image saved to {crop_dst_path}")

    elif label == 20:
        dst_path = os.path.join(output_dir_label20, os.path.basename(src_path))
        shutil.copy(src_path, dst_path)
        print(f"Copied {src_path} to {dst_path}")

        crop_dst_path = os.path.join(crop_dir_label20, os.path.basename(src_path))
        cropped_img.save(crop_dst_path)
        print(f"Cropped image saved to {crop_dst_path}")

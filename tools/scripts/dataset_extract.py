import os
import shutil
from src.dataset import NACTIAnnotationDataset
dataset = NACTIAnnotationDataset(
    image_dir=r"F:\DATASET\NACTI\images",
    json_path=r"E:\result\json\detection\detection_filtered.json",
    csv_path=r"F:/DATASET/NACTI/meta/nacti_metadata_balanced.csv"
)

output_dir_label47 = r"E:\result\nacti_image\wildturkey"
output_dir_label20 = r"E:\result\nacti_image\horse"

os.makedirs(output_dir_label47, exist_ok=True)
os.makedirs(output_dir_label20, exist_ok=True)

for sample in dataset.samples:
    rel_path = sample["rel_path"]
    target = sample["target"]

    label = target["labels"][0].item()

    src_path = os.path.join(dataset.image_dir, rel_path)

    if label == 46:
        dst_path = os.path.join(output_dir_label47, os.path.basename(src_path))
        shutil.copy(src_path, dst_path)
        print(f"Copied {src_path} to {dst_path}")
    elif label == 20:
        dst_path = os.path.join(output_dir_label20, os.path.basename(src_path))
        shutil.copy(src_path, dst_path)
        print(f"Copied {src_path} to {dst_path}")

import os
import json
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class NACTIAnnotationDataset(Dataset):
    def __init__(self, image_dir, json_path, csv_path, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.annotations = data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in data['categories']}
        self.image_id_to_file = {img['id']: img['file_name'] for img in data.get('images', [])}

        self.csv_data = pd.read_csv(csv_path)
        self.csv_data['filename'] = self.csv_data['filename'].apply(lambda x: x.split('/', 2)[-1])
        self.filename_to_common_name = dict(zip(self.csv_data['filename'], self.csv_data['common_name']))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        image_id = annotation['image_id']
        img_filename = self.image_id_to_file.get(image_id, f"{image_id}.jpg")

        img_filename_cleaned = img_filename.split('/', 2)[-1]
        img_path = os.path.join(self.image_dir, img_filename_cleaned)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at: {img_path}")

        image = Image.open(img_path).convert("RGB")

        bbox = annotation['bbox']  # [x_min, y_min, width, height]
        label = annotation['category_id']

        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]

        common_name = self.filename_to_common_name.get(img_filename_cleaned, "unknown")

        target = {
            "boxes": torch.tensor([bbox], dtype=torch.float32),
            "labels": torch.tensor([label], dtype=torch.int64),
            "common_name": common_name
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

dataset = NACTIAnnotationDataset(
    image_dir=r"F:\DATASET\NACTI\images\nacti_part0",
    json_path=r"F:\DATASET\NACTI\meta\nacti_20230920_bboxes.json",
    csv_path=r"F:\DATASET\NACTI\meta\nacti_metadata_part0.csv"
)

for idx in range(5):
    try:
        image, target = dataset[idx]
        print(f"Image loaded: {image.size}, Target: {target}")
    except FileNotFoundError as e:
        print(e)

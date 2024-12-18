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
        # self.categories = {cat['id']: cat['name'] for cat in data['categories']}
        # self.image_id_to_file = {img['id']: img['file_name'] for img in data.get('images', [])}

        self.csv_data = pd.read_csv(csv_path)
        self.csv_data['filename'] = self.csv_data['filename'].apply(lambda x: x.split('/', 2)[-1])
        self.filename_to_common_name = dict(zip(self.csv_data['filename'], self.csv_data['common_name']))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        img_path = annotation['img_id']
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at: {img_path}")

        image = Image.open(img_path).convert("RGB")

        # Handle missing or empty bbox
        if not annotation.get('bbox') or len(annotation['bbox']) == 0:
            # raise ValueError(f"Missing or empty bbox in annotation: {annotation}")
            # just skip this annotation
            return None

        bbox = annotation['bbox'][0]
        label = annotation.get('category', [0])[0]  # Default to category 0 if missing

        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        img_filename = os.path.basename(img_path)
        common_name = self.filename_to_common_name.get(img_filename, "unknown")

        target = {
            "boxes": torch.tensor([bbox], dtype=torch.float32),
            "labels": torch.tensor([label], dtype=torch.int64),
            "common_name": common_name
        }

        # Apply transforms if provided
        if self.transforms:
            image = self.transforms(image)

        return image, target


dataset = NACTIAnnotationDataset(
    image_dir=r"F:\DATASET\NACTI\images\nacti_part0",
    json_path=r"E:\result\json\detection\part0output.json",
    csv_path=r"F:\DATASET\NACTI\meta\nacti_metadata_part0.csv"
)

for idx in range(5):
    try:
        image, target = dataset[idx]
        print(f"Image loaded: {image.size}, Target: {target}")
    except FileNotFoundError as e:
        print(e)

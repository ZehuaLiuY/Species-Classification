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
        self.annotations = data['annotations']  # structure: list of dict
        # self.categories = {cat['id']: cat['name'] for cat in data['categories']}
        # self.image_id_to_file = {img['id']: img['file_name'] for img in data.get('images', [])}

        self.csv_data = pd.read_csv(csv_path)
        self.csv_data['filename'] = self.csv_data['filename'].apply(lambda x: x.split('/', 2)[-1])
        self.filename_to_common_name = dict(zip(self.csv_data['filename'], self.csv_data['common_name']))

        # get all common names and mapping to int
        all_common_names = sorted(set(self.csv_data['common_name']))
        self.common_name_to_idx = {cn: i for i, cn in enumerate(all_common_names)}
        print("common_name_to_idx:", self.common_name_to_idx)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = annotation['img_id']

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at: {img_path}")

        image = Image.open(img_path).convert("RGB")

        # using filename to get common_name
        img_filename = os.path.basename(img_path)
        common_name = self.filename_to_common_name.get(img_filename, "unknown")
        #
        if common_name == "unknown" or common_name not in self.common_name_to_idx:
            return None

        # get mapping the common_name to int
        label_idx = self.common_name_to_idx[common_name]

        if not annotation.get('bbox') or len(annotation['bbox']) == 0:
            return None
        bboxes = annotation.get('bbox', [])
        if len(bboxes) == 0:
            return None

        xywh_boxes = []
        for b in bboxes:
            x1, y1, x2, y2 = b
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            xywh_boxes.append([x1, y1, x2 - x1, y2 - y1])

        if len(xywh_boxes) == 0:
            return None

        labels = [label_idx] * len(xywh_boxes)

        target = {
            "boxes": torch.tensor(xywh_boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "common_name": common_name
        }

        # transforms
        if self.transforms:
            image = self.transforms(image)

        return image, target


# testing the dataset
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

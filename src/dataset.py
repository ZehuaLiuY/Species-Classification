import os
import json
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class NACTIAnnotationDataset(Dataset):
    def __init__(self, image_dir, json_path, csv_path, transforms=None, allow_empty=False):
        """
        :param image_dir: The root directory for the images, used to construct the full image path.
        :param json_path: Path to the JSON file containing 'annotations', which is a list of dicts.
        :param csv_path: Path to a CSV file that contains at least 'id', 'filename', (optionally) 'common_name' columns.
        :param transforms: (Optional) image transformations or augmentations.
        :param allow_empty: Whether to allow samples with no valid bounding boxes (treated as negative samples).
        """
        self.image_dir = image_dir
        self.transforms = transforms
        self.allow_empty = allow_empty  # If True, samples with no valid bounding boxes can still be included.

        # 1) Load the CSV file and build a mapping from the basename (id) to the relative filename.
        csv_data = pd.read_csv(csv_path)
        # Replace backslashes with forward slashes for consistency on Windows.
        csv_data['filename'] = csv_data['filename'].apply(lambda x: x.replace('\\', '/'))

        # If the CSV has a 'common_name' column, store the values in a dictionary: basename -> common_name.
        self.id_to_common_name = {}
        if 'common_name' in csv_data.columns:
            for _, row in csv_data.iterrows():
                basename = row['id']   # e.g., "CA-39_0003179.jpg"
                cname = row['common_name']
                self.id_to_common_name[basename] = cname

        # Build a dictionary for basename -> relative_path.
        self.id_to_filename = dict(zip(csv_data['id'], csv_data['filename']))

        # 2) Read the JSON file and build a temporary dictionary: rel_path -> list of detection entries.
        with open(json_path, 'r') as f:
            data = json.load(f)
        annotations = data['annotations']  # a list of dicts

        self.filename_to_anns = {}
        for ann in annotations:
            raw_path = ann['img_id'].replace('\\', '/')
            base = os.path.basename(raw_path)  # Extract something like "CA-39_0003179.jpg"

            # If the CSV does not have an entry for this base, skip it.
            if base not in self.id_to_filename:
                continue

            rel_path = self.id_to_filename[base]
            if rel_path not in self.filename_to_anns:
                self.filename_to_anns[rel_path] = []
            # Append the detection data (possibly multiple bounding boxes in "bbox").
            self.filename_to_anns[rel_path].append({
                "bbox": ann.get("bbox", []),
                "category": ann.get("category", []),
                "confidence": ann.get("confidence", [])
            })

        # 3) In __init__, parse bounding boxes, apply filtering, and build self.samples.
        #    Each element in self.samples is a dict that includes the relative path and the pre-processed target.
        self.samples = []

        # Iterate through all relative paths in filename_to_anns
        for rel_path, ann_list in self.filename_to_anns.items():
            all_boxes = []
            all_labels = []
            all_confs = []

            # Merge all detection data for this rel_path
            for det_info in ann_list:
                for bbox in det_info["bbox"]:
                    x1, y1, x2, y2 = bbox
                    # Convert to (x, y, w, h)
                    w = x2 - x1
                    h = y2 - y1
                    # Filter out invalid bounding boxes (w <= 0 or h <= 0) if necessary
                    if w > 0 and h > 0:
                        all_boxes.append([x1, y1, w, h])

                cat_list = det_info["category"]
                conf_list = det_info["confidence"]
                all_labels.extend(cat_list)
                all_confs.extend(conf_list)

            # If there are no valid boxes and we do NOT allow empty, skip this sample.
            if len(all_boxes) == 0 and not self.allow_empty:
                continue

            # Construct tensors for boxes, labels, and confidences
            boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32) if len(all_boxes) else torch.empty((0,4), dtype=torch.float32)
            labels_tensor = torch.tensor(all_labels, dtype=torch.int64) if len(all_labels) else torch.empty((0,), dtype=torch.int64)
            confs_tensor = torch.tensor(all_confs, dtype=torch.float32) if len(all_confs) else torch.empty((0,), dtype=torch.float32)

            sample_target = {
                "boxes": boxes_tensor,
                "labels": labels_tensor,
                "scores": confs_tensor
            }

            # 4) If we have a common_name, retrieve it from the dictionary.
            base = os.path.basename(rel_path)
            if base in self.id_to_common_name:
                cname = self.id_to_common_name[base]
                # If we also want to filter out samples where common_name == "empty", do it here:
                if cname.lower() == "empty":
                    continue
                sample_target["common_name"] = cname

            # Finally, add (rel_path, target) to self.samples
            self.samples.append({
                "rel_path": rel_path,
                "target": sample_target
            })

        # The final dataset length is the length of self.samples.
        print(f"[NACTIAnnotationDataset] Constructed {len(self.samples)} samples after filtering.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Here we do not parse bounding boxes or filter anymore; we simply load the image and return the preprocessed target.
        sample = self.samples[idx]
        rel_path = sample["rel_path"]
        target = sample["target"]

        img_path = os.path.join(self.image_dir, rel_path)
        image = Image.open(img_path).convert("RGB")

        # Apply transforms if provided
        if self.transforms:
            image = self.transforms(image)

        return image, target


# Testing the dataset
dataset = NACTIAnnotationDataset(
    image_dir=r"F:\DATASET\NACTI\images",
    json_path=r"E:\result\json\detection\part0output.json",
    csv_path=r"F:\DATASET\NACTI\meta\nacti_metadata_part0.csv"
)

# Check the length of the dataset
print("Dataset length:", len(dataset))

for idx in range(5):
    try:
        image, target = dataset[idx]
        print(f"Image loaded: {image.size}, Target: {target}")
    except FileNotFoundError as e:
        print(e)


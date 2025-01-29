import os
import json
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class NACTIAnnotationDataset(Dataset):
    def __init__(self, image_dir, json_path, csv_path, transforms=None, allow_empty=False):
        """
        :param image_dir: Root directory to construct the full absolute path of images
        :param json_path: JSON file containing the 'annotations' list
        :param csv_path: CSV file containing at least 'id', 'filename', and optionally 'common_name' columns
        :param transforms: Image transformations/augmentations
        :param allow_empty: Whether to allow images without valid bounding boxes to be included (as negative samples)
        """
        self.image_dir = image_dir
        self.transforms = transforms
        self.allow_empty = allow_empty  # If True, retain images with empty/invalid annotations

        # 1) Read the CSV to construct a mapping from basename -> filename
        csv_data = pd.read_csv(csv_path)
        # Normalize slashes
        csv_data['filename'] = csv_data['filename'].apply(lambda x: x.replace('\\', '/'))

        # If the CSV contains a 'common_name' column, store it in a dictionary
        self.id_to_common_name = {}
        if 'common_name' in csv_data.columns:
            for _, row in csv_data.iterrows():
                basename = row['id']   # e.g., "CA-39_0003179.jpg"
                cname = row['common_name']
                self.id_to_common_name[basename] = cname

        # Construct a mapping from basename -> relative_path
        self.id_to_filename = dict(zip(csv_data['id'], csv_data['filename']))

        # 2) Read the JSON to construct a temporary dictionary: rel_path -> [list of detection information]
        with open(json_path, 'r') as f:
            data = json.load(f)
        annotations = data['annotations']  # list of dicts

        self.filename_to_anns = {}  # Temporarily store [detection records...]
        for ann in annotations:
            raw_path = ann['img_id'].replace('\\', '/')
            base = os.path.basename(raw_path)  # Extract "CA-39_0003179.jpg"

            # If the base is not found in the CSV, it cannot be matched
            if base not in self.id_to_filename:
                continue

            rel_path = self.id_to_filename[base]
            if rel_path not in self.filename_to_anns:
                self.filename_to_anns[rel_path] = []
            self.filename_to_anns[rel_path].append({
                "bbox": ann.get("bbox", []),
                "category": ann.get("category", []),
                "confidence": ann.get("confidence", [])
            })

        # 3) Parse and filter bounding box information in the init method, then save to self.samples
        #    Each element can be a dict containing all the necessary information for an image
        self.samples = []  # Later, __getitem__ will retrieve data from here

        # Iterate through all rel_path entries
        for rel_path, ann_list in self.filename_to_anns.items():
            all_boxes = []
            all_labels = []
            all_confs = []

            # Merge multiple annotations
            for det_info in ann_list:
                for bbox in det_info["bbox"]:
                    x1, y1, x2, y2 = bbox
                    # Convert to (x, y, w, h)
                    w = x2 - x1
                    h = y2 - y1
                    # To filter out invalid bounding boxes (w <= 0 or h <= 0), add a condition here
                    if w > 0 and h > 0:
                        all_boxes.append([x1, y1, w, h])

                cat_list = det_info["category"]
                conf_list = det_info["confidence"]
                all_labels.extend(cat_list)
                all_confs.extend(conf_list)

            # If there are no valid bounding boxes after filtering:
            if len(all_boxes) == 0:
                # Retain based on allow_empty
                if not self.allow_empty:
                    # Skip this sample
                    continue

            # Create tensors
            boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32) if len(all_boxes) else torch.empty((0,4), dtype=torch.float32)
            labels_tensor = torch.tensor(all_labels, dtype=torch.int64) if len(all_labels) else torch.empty((0,), dtype=torch.int64)
            confs_tensor = torch.tensor(all_confs, dtype=torch.float32) if len(all_confs) else torch.empty((0,), dtype=torch.float32)

            # Assemble the target
            sample_target = {
                "boxes": boxes_tensor,
                "labels": labels_tensor,
                "scores": confs_tensor
            }

            # If a common_name exists
            base = os.path.basename(rel_path)
            if base in self.id_to_common_name:
                sample_target["common_name"] = self.id_to_common_name[base]

            # Add the final (rel_path, target) to samples
            self.samples.append({
                "rel_path": rel_path,
                "target": sample_target
            })

        # 4) The final dataset length is the length of self.samples
        print(f"[NACTIAnnotationDataset] Constructed {len(self.samples)} samples after filtering.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # No further parsing or filtering; directly retrieve pre-processed data
        sample = self.samples[idx]
        rel_path = sample["rel_path"]
        target = sample["target"]

        img_path = os.path.join(self.image_dir, rel_path)
        image = Image.open(img_path).convert("RGB")

        # Apply transforms if needed
        if self.transforms:
            image = self.transforms(image)

        return image, target

# Testing the dataset
dataset = NACTIAnnotationDataset(
    image_dir=r"F:\DATASET\NACTI\images",
    json_path=r"E:\result\json\detection\part3output.json",
    csv_path=r"F:\DATASET\NACTI\meta\nacti_metadata_part3.csv"
)

# Check the length of the dataset
print("Dataset length:", len(dataset))

for idx in range(5):
    try:
        image, target = dataset[idx]
        print(f"Image loaded: {image.size}, Target: {target}")
    except FileNotFoundError as e:
        print(e)

# Common name mapping:
# common_name_to_idx: {
# 'american black bear': 0,
# 'american marten': 1,
# 'american red squirrel': 2,
# 'black-tailed jackrabbit': 3,
# 'bobcat': 4,
# 'california ground squirrel': 5,
# 'california quail': 6,
# 'cougar': 7,
# 'coyote': 8,
# 'dark-eyed junco': 9,
# 'domestic cow': 10,
# 'domestic dog': 11,
# 'dusky grouse': 12,
# 'eastern gray squirrel': 13,
# 'elk': 14,
# 'ermine': 15,
# 'european badger': 16,
# 'gray fox': 17,
# 'gray jay': 18,
# 'horse': 19,
# 'house wren': 20,
# 'long-tailed weasel': 21,
# 'moose': 22,
# 'mule deer': 23,
# 'north american porcupine': 24,
# 'raccoon': 25,
# 'red deer': 26,
# 'red fox': 27,
# 'snowshoe hare': 28,
# "steller's jay": 29,
# 'striped skunk': 30,
# 'unidentified accipitrid': 31,
# 'unidentified bird': 32,
# 'unidentified chipmunk': 33,
# 'unidentified corvus': 34,
# 'unidentified deer': 35,
# 'unidentified deer mouse': 36,
# 'unidentified mouse': 37,
# 'unidentified pack rat': 38,
# 'unidentified pocket gopher': 39,
# 'unidentified rabbit': 40,
# 'vehicle': 41,
# 'virginia opossum': 42,
# 'wild boar': 43,
# 'wild turkey': 44,
# 'yellow-bellied marmot': 45
# }

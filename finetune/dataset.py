import os
import json
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

class NACTIAnnotationDataset(Dataset):
    def __init__(self, image_dir, json_path, csv_path, transforms=None, allow_empty=False):
        """
        :param image_dir: Root directory for the images.
        :param json_path: Path to the JSON file with 'annotations', which is a list of dicts.
        :param csv_path:  Path to a CSV file containing at least 'id', 'filename', (optionally) 'common_name' columns.
        :param transforms: (Optional) Image transformations or augmentations.
        :param allow_empty: Whether to allow samples with no valid bounding boxes (treated as negative samples).
        """
        self.image_dir = image_dir
        self.transforms = transforms
        self.allow_empty = allow_empty

        # --------------------------------------------------------------
        # 1) Read CSV file and create mappings:
        #    - id_to_filename: maps the basename (e.g. "CA-39_0003179.jpg") to its relative file path.
        #    - id_to_common_name: maps basename to its common_name string.
        #    - common_name_to_id: maps each unique common_name to an integer index.
        # --------------------------------------------------------------
        csv_data = pd.read_csv(csv_path, low_memory=False)

        # Ensure forward slashes on Windows systems
        csv_data['filename'] = csv_data['filename'].apply(lambda x: x.replace('\\', '/'))

        # Build a mapping from id -> filename
        self.id_to_filename = dict(zip(csv_data['id'], csv_data['filename']))

        # If there is a common_name column, collect unique names and build a mapping to integer IDs
        self.id_to_common_name = {}
        self.common_name_to_id = {}
        if 'common_name' in csv_data.columns:
            # Create a set (or list) of unique common_names
            unique_cnames = csv_data['common_name'].dropna().unique().tolist()

            # delete the empty string if the allow_empty is False
            if not allow_empty and 'empty' in unique_cnames:
                unique_cnames.remove('empty')
                unique_cnames.remove('vehicle')


            unique_cnames = sorted(unique_cnames)
            # Build a dictionary mapping each unique common_name to an integer
            self.common_name_to_id = {cname: idx for idx, cname in enumerate(unique_cnames)}
            print(f"[NACTIAnnotationDataset] Found {len(unique_cnames)} unique common names.")
            # print(f"[NACTIAnnotationDataset] Common names: {unique_cnames}")
            # print(self.common_name_to_id)

            # Also store id -> common_name for each row
            for _, row in csv_data.iterrows():
                basename = row['id']    # e.g. "CA-39_0003179.jpg"
                cname = row['common_name']
                self.id_to_common_name[basename] = cname

        # --------------------------------------------------------------
        # 2) Parse the JSON file (detection results). For each 'img_id',
        #    we collect the bounding boxes and detection info in a dict:
        #    filename_to_anns[rel_path] -> list of detection entries
        # --------------------------------------------------------------
        with open(json_path, 'r') as f:
            data = json.load(f)
        annotations = data['annotations']  # a list of dicts

        self.filename_to_anns = {}
        for ann in annotations:
            # Normalize path separators
            raw_path = ann['img_id'].replace('\\', '/')
            base = os.path.basename(raw_path)  # e.g. "CA-39_0003179.jpg"

            # If we don't have a corresponding entry in CSV, skip
            if base not in self.id_to_filename:
                continue

            rel_path = self.id_to_filename[base]
            if rel_path not in self.filename_to_anns:
                self.filename_to_anns[rel_path] = []

            # Each annotation may contain multiple bounding boxes in ann["bbox"]
            self.filename_to_anns[rel_path].append({
                "bbox": ann.get("bbox", []),
                # We will NOT use ann['category'] as labels, ignoring it here.
                "confidence": ann.get("confidence", [])
            })

        # --------------------------------------------------------------
        # 3) Build self.samples by merging bounding boxes per image
        #    and assigning labels from CSV's common_name (if available).
        # --------------------------------------------------------------
        self.samples = []

        for rel_path, ann_list in self.filename_to_anns.items():
            all_boxes = []
            all_confs = []

            # Merge bounding boxes for this rel_path
            for det_info in ann_list:
                for bbox in det_info["bbox"]:
                    x1, y1, x2, y2 = bbox
                    w = x2 - x1
                    h = y2 - y1
                    # Filter out invalid bounding boxes
                    if w > 0 and h > 0:
                        all_boxes.append([x1, y1, w, h])

                conf_list = det_info["confidence"]
                all_confs.extend(conf_list)

            # If no valid boxes and not allow_empty, skip
            if len(all_boxes) == 0 and not self.allow_empty:
                continue

            # Convert to tensors
            boxes_tensor = (torch.tensor(all_boxes, dtype=torch.float32)
                            if len(all_boxes) else torch.empty((0,4), dtype=torch.float32))
            confs_tensor = (torch.tensor(all_confs, dtype=torch.float32)
                            if len(all_confs) else torch.empty((0,), dtype=torch.float32))

            # -----------------------
            # Retrieve the common_name for this image. If it's missing or "empty",
            # or not found in common_name_to_id, we skip.
            # -----------------------
            base = os.path.basename(rel_path)
            if base not in self.id_to_common_name:
                # Skip if there's no known common_name for this image
                continue

            cname = self.id_to_common_name[base]
            if isinstance(cname, str) and cname.lower() == "empty":
                continue
            if cname not in self.common_name_to_id:
                # Skip if the CSV's common_name is not in our mapping
                continue

            # Build the label tensor:
            class_id = self.common_name_to_id[cname]
            # All bounding boxes in this image get the same class_id
            labels_tensor = torch.tensor([class_id]*len(all_boxes), dtype=torch.int64)

            # Create the target dict
            sample_target = {
                "boxes": boxes_tensor,
                "labels": labels_tensor,
                "scores": confs_tensor,
                "common_name": cname  # Optionally keep the string label as well
            }

            # Save this sample
            self.samples.append({
                "rel_path": rel_path,
                "target": sample_target
            })

        print(f"[NACTIAnnotationDataset] Constructed {len(self.samples)} samples after filtering.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return:
           (PIL.Image or transformed Tensor, target_dict)
        """
        sample = self.samples[idx]
        rel_path = sample["rel_path"]
        target = sample["target"]

        # Load the image
        img_path = os.path.join(self.image_dir, rel_path)
        image = Image.open(img_path).convert("RGB")

        # Apply transforms if provided
        if self.transforms:
            image = self.transforms(image)

        return image, target


# # Testing the dataset
# dataset = NACTIAnnotationDataset(
#     image_dir=r"F:\DATASET\NACTI\images",
#     json_path=r"E:\result\json\detection\detection_filtered.json",
#     csv_path=r"F:/DATASET/NACTI/meta/nacti_metadata_balanced.csv"
# )
#
# # Check the length of the dataset
# print("Dataset length:", len(dataset))
#
# for idx in range(5):
#     try:
#         image, target = dataset[idx]
#         print(f"Image loaded: {image.size}, Target: {target}")
#     except FileNotFoundError as e:
#         print(e)


class ReducedBiasedDataset(Dataset):
    def __init__(self, image_dir, json_path, transforms=None, allow_empty=False):
        self.image_dir = image_dir
        self.transforms = transforms
        self.allow_empty = allow_empty

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.id_to_filename = {}
        self.id_to_size = {}

        for img in data["images"]:
            img_id = str(img["id"])
            self.id_to_filename[img_id] = os.path.normpath(img["file_name"])
            self.id_to_size[img_id] = (img.get("width", 0), img.get("height", 0))

        self.categories = {}
        if "categories" in data:
            for cat in data["categories"]:
                self.categories[cat["id"]] = cat["name"]

        self.filename_to_anns = {}
        for ann in data["annotations"]:
            img_id = str(ann["image_id"])
            if img_id not in self.id_to_filename:
                continue
            file_name = self.id_to_filename[img_id]
            if file_name not in self.filename_to_anns:
                self.filename_to_anns[file_name] = []
            bbox = ann.get("bbox", [])

            if len(bbox) != 4:
                w, h = self.id_to_size.get(img_id, (0, 0))
                if w > 0 and h > 0:
                    bbox = [0, 0, w, h]

            if len(bbox) == 4:
                x, y, w, h = bbox
                if w > 0 and h > 0:
                    label = ann.get("category_id", -1)
                    score = ann.get("score", 1.0)
                    self.filename_to_anns[file_name].append({
                        "bbox": [x, y, w, h],
                        "score": score,
                        "label": label,
                        "category_name": self.categories.get(label, "") if label != -1 else ""
                    })

        self.samples = []
        for file_name, ann_list in self.filename_to_anns.items():
            boxes = []
            labels = []
            scores = []
            for ann in ann_list:
                boxes.append(ann["bbox"])
                labels.append(ann["label"])
                scores.append(ann["score"])

            if len(boxes) == 0 and not self.allow_empty:
                continue

            boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)
            scores_tensor = torch.tensor(scores, dtype=torch.float32) if scores else torch.empty((0,), dtype=torch.float32)

            target = {
                "boxes": boxes_tensor,
                "labels": labels_tensor,
                "scores": scores_tensor
            }
            self.samples.append({
                "file_name": file_name,
                "target": target
            })

        print(f"[Dataset] Constructed {len(self.samples)} samples after filtering.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        file_name = sample["file_name"]
        target = sample["target"]

        img_path = os.path.join(self.image_dir, file_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: {img_path} not found, skipping sample {idx}")
            return (None, None)

        if self.transforms:
            image = self.transforms(image)

        return image, target
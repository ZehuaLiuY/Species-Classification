import torch
import pandas as pd
from torch.utils.data import Dataset
from dataset import NACTIAnnotationDataset
from torchvision import transforms
from transformers import BertTokenizer

class MultimodalNACTIDataset(Dataset):
    def __init__(self, image_dir, json_path, csv_path, text_csv_path=None, text_mapping=None,
                 transforms=None, allow_empty=False, tokenizer=None, max_length=128, crop_transform=None):
        self.base_dataset = NACTIAnnotationDataset(image_dir, json_path, csv_path,
                                                   transforms=transforms, allow_empty=allow_empty)

        # load text mapping
        if text_mapping is not None:
            self.text_mapping = text_mapping
        elif text_csv_path is not None:
            df = pd.read_csv(text_csv_path)
            df.columns = df.columns.str.strip()
            df['common_name'] = df['common_name'].str.lower()
            # key: common_name, value: description
            self.text_mapping = dict(zip(df['common_name'], df['description']))
        else:
            # if no text mapping provided, use common_name as text
            self.text_mapping = {}

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer_cache = {}
        self.crop_transform = crop_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # get image, text and target from base dataset
        image, target = self.base_dataset[idx]
        common_name = target.get("common_name", "").lower()
        text_str = self.text_mapping.get(common_name, common_name)
        if self.tokenizer is not None:
            if text_str not in self.tokenizer_cache:
                tokenized = self.tokenizer(text_str, max_length=self.max_length,
                                           padding='max_length', truncation=True, return_tensors='pt')
                tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
                self.tokenizer_cache[text_str] = tokenized
            tokenized_text = self.tokenizer_cache[text_str]
        else:
            tokenized_text = text_str

        # get crops and labels
        boxes = target.get("boxes", [])
        labels = target.get("labels", [])
        crops = []
        crop_labels = []
        # if boxes is a tensor, convert it to list
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.tolist()
        for box, label in zip(boxes, labels):
            # crop image
            x, y, w, h = box
            x2 = x + w
            y2 = y + h
            cropped_img = image.crop((int(x), int(y), int(x2), int(y2)))
            if self.crop_transform is not None:
                cropped_img = self.crop_transform(cropped_img)
            crops.append(cropped_img)
            if isinstance(label, torch.Tensor):
                label = label.item()
            crop_labels.append(label)
        return crops, tokenized_text, crop_labels

# collate_fn for DataLoader
def multimodal_collate_fn(batch):
    all_crops = []
    all_tokenized_texts = []
    all_labels = []
    for crops, tokenized_text, crop_labels in batch:
        for crop, label in zip(crops, crop_labels):
            all_crops.append(crop)
            all_labels.append(label)
            all_tokenized_texts.append(tokenized_text)
    return all_crops, all_tokenized_texts, all_labels

if __name__ == "__main__":
    image_dir = r"F:\DATASET\NACTI\images"
    json_path = r"E:\result\json\detection\detection_filtered.json"
    csv_path = r"F:/DATASET/NACTI/meta/nacti_metadata_balanced.csv"
    text_csv_path = r"F:/DATASET/NACTI/meta/description.csv"

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


    crop_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    multimodal_dataset = MultimodalNACTIDataset(
        image_dir=image_dir,
        json_path=json_path,
        csv_path=csv_path,
        text_csv_path=text_csv_path,
        transforms=None,
        allow_empty=False,
        tokenizer=tokenizer,
        max_length=128,
        crop_transform=crop_transform
    )

    crops, tokenized_text, crop_labels = multimodal_dataset[0]
    print("Number of cropped regions:", len(crops))
    if len(crops) > 0:
        print("Crop tensor shape:", crops[0].shape)
    print("Tokenized text:", tokenized_text)
    print("Labels:", crop_labels)

import os
import json
import torch
from PIL import Image
from torchvision import datasets, transforms
from PytorchWildlife.models import classification as pw_classification
from tqdm import tqdm
from torch.utils.data import random_split
from src.dataset import NACTIAnnotationDataset
from torch.utils.data import DataLoader
import torch.optim as optim

json_file = r'E:\result\json\detection\part1output.json'
image_folder = r'F:\DATASET\NACTI\images\nacti_part0'

# Transform for classification
transform = transforms.Compose([
    transforms.Resize((1280, 1280)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classification_model = pw_classification.AI4GAmazonRainforest(device=device)

# split the dataset into training and validation part 80% and 20%
dataset = NACTIAnnotationDataset(
    image_dir=r"F:\DATASET\NACTI\images\nacti_part0",
    json_path=r"E:\result\json\detection\part0output.json",
    csv_path=r"F:\DATASET\NACTI\meta\nacti_metadata_part0.csv",
    transforms=transform
)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# fine-tune the model
classification_model.train()
classification_model.to(device)
optimizer = optim.AdamW(classification_model.parameters(), lr=0.0001)
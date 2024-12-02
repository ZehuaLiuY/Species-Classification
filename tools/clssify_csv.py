import os
import json
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2
from torch.utils.data import DataLoader
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife.data.datasets import DetectionImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Pretrained model classes
CLASS_NAMES = {0: 'Dasyprocta', 1: 'Bos', 2: 'Pecari', 3: 'Mazama', 4: 'Cuniculus', 
               5: 'Leptotila', 6: 'Human', 7: 'Aramides', 8: 'Tinamus', 9: 'Eira', 
               10: 'Crax', 11: 'Procyon', 12: 'Capra', 13: 'Dasypus', 14: 'Sciurus', 
               15: 'Crypturellus', 16: 'Tamandua', 17: 'Proechimys', 18: 'Leopardus', 
               19: 'Equus', 20: 'Columbina', 21: 'Nyctidromus', 22: 'Ortalis', 
               23: 'Emballonura', 24: 'Odontophorus', 25: 'Geotrygon', 26: 'Metachirus', 
               27: 'Catharus', 28: 'Cerdocyon', 29: 'Momotus', 30: 'Tapirus', 31: 'Canis', 
               32: 'Furnarius', 33: 'Didelphis', 34: 'Sylvilagus', 35: 'Unknown'}

# Paths
image_folder = r'G:\result\detected'
output_folder = r'G:\result\cropped_images'
json_file = r'E:\result\json\detection\part3output.json'
classification_results_csv = 'classification_results.csv'


# Load detections
with open(json_file) as f:
    detections = json.load(f)

# Transform for classification
transform = transforms.Compose([
    transforms.Resize((1280, 1280)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset and dataloader
dataset = DetectionImageFolder(image_dir=output_folder, transform=transform)
print(f"Loaded dataset with {len(dataset)} images")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

# Load model
classification_model = pw_classification.AI4GAmazonRainforest(device=device)
print("Model loaded")

# Classification and saving results
classification_results = []  # List to store classification results

with torch.no_grad():
    for imgs, img_paths, _ in dataloader:
        imgs = imgs.to(classification_model.device)
        print(f"Loaded batch with {len(imgs)} images")
        logits = classification_model.forward(imgs)
        probs = torch.softmax(logits, dim=1)
        max_probs, predicted_classes = torch.max(probs, dim=1)

        for img_path, predicted_class, prob in zip(img_paths, predicted_classes, max_probs):
            classification_results.append({
                'filename': os.path.basename(img_path),
                'predicted_class': predicted_class.item(),
                'probability': prob.item()
            })
            print(f"Cropped image from {img_path} classified as: {predicted_class.item()} with probability: {prob.item():.2f}")

# Save classification results to JSON
classification_results_json =  r'G:\result\E:\result\json\classification\part0.json'
with open(classification_results_json, 'w', encoding='utf-8') as json_file:
    json.dump(classification_results, json_file, indent=4, ensure_ascii=False)
print(f"Classification results saved to {classification_results_json}")

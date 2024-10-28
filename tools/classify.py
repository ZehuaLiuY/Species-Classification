import os
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2
from torch.utils.data import DataLoader
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife.data.datasets import DetectionImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# pretrained model classes
CLASS_NAMES = {0: 'Dasyprocta', 1: 'Bos', 2: 'Pecari', 3: 'Mazama', 4: 'Cuniculus', 5: 'Leptotila', 6: 'Human', 7: 'Aramides', 8: 'Tinamus', 9: 'Eira', 10: 'Crax', 11: 'Procyon', 12: 'Capra', 13: 'Dasypus', 14: 'Sciurus', 15: 'Crypturellus', 16: 'Tamandua', 17: 'Proechimys', 18: 'Leopardus', 19: 'Equus', 20: 'Columbina', 21: 'Nyctidromus', 22: 'Ortalis', 23: 'Emballonura', 24: 'Odontophorus', 25: 'Geotrygon', 26: 'Metachirus', 27: 'Catharus', 28: 'Cerdocyon', 29: 'Momotus', 30: 'Tapirus', 31: 'Canis', 32: 'Furnarius', 33: 'Didelphis', 34: 'Sylvilagus', 35: 'Unknown'}

# paths
image_folder = r'G:\result\detected'
output_folder = r'G:\result\cropped_images'
labeled_folder = r'G:\result\labeled_images'
json_file = 'part3output.json'
speicies_folder = r'G:\result\species'

if not os.path.exists(labeled_folder):
    os.makedirs(labeled_folder)
if not os.path.exists(speicies_folder):
    os.makedirs(speicies_folder)

with open(json_file) as f:
    detections = json.load(f)

# transform for classification
transform = transforms.Compose([
    transforms.Resize((1280, 1280)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# dataset
dataset = DetectionImageFolder(image_dir=output_folder, transform=transform)
print(f"Loaded dataset with {len(dataset)} images")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
# model
classification_model = pw_classification.AI4GAmazonRainforest(device=device)
print("Model loaded")
# classification
classification_results = {}  # format: {'img_path': ('class_name', probability)}
with torch.no_grad():
    for imgs, img_paths, _ in dataloader:
        imgs = imgs.to(classification_model.device)
        print(f"Loaded batch with {len(imgs)} images")
        logits = classification_model.forward(imgs)
        probs = torch.softmax(logits, dim=1)
        max_probs, predicted_classes = torch.max(probs, dim=1)

        for img_path, predicted_class, prob in zip(img_paths, predicted_classes, max_probs):
            classification_results[os.path.basename(img_path)] = (predicted_class.item(), prob.item())
            print(f"Cropped image from {img_path} classified as: {predicted_class.item()} with probability: {prob.item():.2f}")

# annotate images
for image_data in detections['images']:
    image_path = os.path.join(image_folder, image_data['file'])
    image = cv2.imread(image_path)

    highest_prob = 0
    best_bbox = None
    best_label = "Unknown"

    for idx, detection in enumerate(image_data['detections']):
        # only process animal category
        if detection['category'] == '1':
            bbox = detection['bbox']
            x1, y1, w, h = int(bbox[0] * image.shape[1]), int(bbox[1] * image.shape[0]), \
                int(bbox[2] * image.shape[1]), int(bbox[3] * image.shape[0])
            x2, y2 = x1 + w, y1 + h

            # get classification label
            crop_name = f"{os.path.splitext(os.path.basename(image_data['file']))[0]}_crop_{idx}.jpg"
            predicted_class_id, prob = classification_results.get(crop_name, ("Unknown", 0))
            label = CLASS_NAMES.get(predicted_class_id, "Unknown")

            if prob > highest_prob:
                highest_prob = prob
                best_bbox = (x1, y1, x2, y2)
                best_label = f"{label} ({highest_prob:.2f})"

    if best_bbox:
        x1, y1, x2, y2 = best_bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(image, best_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
        print(f"Annotating {image_data['file']} with label: {best_label}")

    # save annotated image
    labeled_image_path = os.path.join(labeled_folder, os.path.basename(image_data['file']))

    os.makedirs(os.path.dirname(labeled_image_path), exist_ok=True)
    os.makedirs(os.path.dirname(speicies_folder), exist_ok=True)

    # save species images
    if best_label != "Unknown":
        species_image_path = os.path.join(speicies_folder, os.path.basename(image_data['file']))
        cv2.imwrite(species_image_path, image)
        print(f"Saving annotated image: {species_image_path}")
    else:
        # unknown species
        cv2.imwrite(labeled_image_path, image)
        print(f"Saving annotated image: {labeled_image_path}")

print("Finished labeling:", labeled_folder)

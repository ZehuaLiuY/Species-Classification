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

# paths
image_folder = r'G:\result\detected'
output_folder = r'G:\result\cropped_images'
labeled_folder = r'G:\result\labeled_images'
json_file = 'part3output.json'

if not os.path.exists(labeled_folder):
    os.makedirs(labeled_folder)

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
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
# model
classification_model = pw_classification.AI4GAmazonRainforest(device=device)

# classification
classification_results = {}  # format: {'img_path': 'class_name'}
with torch.no_grad():
    for imgs, img_paths, _ in dataloader:
        imgs = imgs.to(classification_model.device)

        logits = classification_model.forward(imgs)
        probs = torch.softmax(logits, dim=1)
        _, predicted_classes = torch.max(probs, dim=1)

        for img_path, predicted_class in zip(img_paths, predicted_classes):
            classification_results[os.path.basename(img_path)] = predicted_class.item()
            print(f"Cropped image from {img_path} classified as: {predicted_class.item()}")

# annotate images
for image_data in detections['images']:
    image_path = os.path.join(image_folder, image_data['file'])
    image = cv2.imread(image_path)

    for idx, detection in enumerate(image_data['detections']):
        if detection['category'] == '1':
            bbox = detection['bbox']
            x1, y1, w, h = int(bbox[0] * image.shape[1]), int(bbox[1] * image.shape[0]), \
                int(bbox[2] * image.shape[1]), int(bbox[3] * image.shape[0])
            x2, y2 = x1 + w, y1 + h

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # get classification label
            crop_name = f"{image_data['file'].split('.')[0]}_crop_{idx}.jpg"
            label = classification_results.get(crop_name, "Unknown")

            # annotate image
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
            print(f"Annotating {image_data['file']} with label: {label}")

    # save annotated image
    labeled_image_path = os.path.join(labeled_folder, image_data['file'])
    cv2.imwrite(labeled_image_path, image)
    print(f"Saving annotated image: {labeled_image_path}")

print("Finished labeling:", labeled_folder)

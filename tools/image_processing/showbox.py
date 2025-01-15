import os
import json
import cv2

image_folder = r'H:/DATASET/NACTI/images/nactiPart3/'
json_file = r'part3output.json'
output_folder = r'G:\result\detected'

os.makedirs(output_folder, exist_ok=True)

with open(json_file) as f:
    data = json.load(f)

labels = {1: 'animal', 2: 'person', 3: 'vehicle'}

# Create a dictionary of all image paths with the basename as key
image_paths = {}
for root, dirs, files in os.walk(image_folder):
    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):  # Adjust for your image formats
            full_path = os.path.join(root, file)
            image_paths[os.path.basename(file)] = full_path

# Iterate over the images in the JSON file
for img_data in data['images']:
    image_file_name = os.path.basename(img_data['file'])

    if image_file_name not in image_paths:
        print(f"Image {image_file_name} does not exist in the dataset.")
        continue

    image_file = image_paths[image_file_name]
    image = cv2.imread(image_file)

    if image is None:
        print(f"Failed to load image {image_file}")
        continue

    detections = img_data.get('detections', [])
    for detection in detections:
        category = detection['category']
        confidence = detection['conf']
        bbox = detection['bbox']

        x1 = int(bbox[0] * image.shape[1])
        y1 = int(bbox[1] * image.shape[0])
        x2 = int((bbox[0] + bbox[2]) * image.shape[1])
        y2 = int((bbox[1] + bbox[3]) * image.shape[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f'{labels[int(category)]}: {confidence:.2f}'
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    output_path = os.path.join(output_folder, image_file_name)
    cv2.imwrite(output_path, image)
    print(f"Saved image with detections to {output_path}")
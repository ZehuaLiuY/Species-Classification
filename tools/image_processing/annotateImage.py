import os
import json
import cv2
import pandas as pd

# Paths
image_folder = r'G:\result\detected'
output_folder = r'G:\result\cropped_images'
labeled_folder = r'G:\result\labeled_images'
species_folder = r'G:\result\species'
json_file = '../result/part3output.json'
classification_results_csv = '../classification_results.csv'

# Load classification results
classification_df = pd.read_csv(classification_results_csv)
classification_results = {
    row['filename']: (row['predicted_class'], row['probability'])
    for _, row in classification_df.iterrows()
}

# Load detections
with open(json_file) as f:
    detections = json.load(f)

# Predefined class names
CLASS_NAMES = {0: 'Dasyprocta', 1: 'Bos', 2: 'Pecari', 3: 'Mazama', 4: 'Cuniculus', 
               5: 'Leptotila', 6: 'Human', 7: 'Aramides', 8: 'Tinamus', 9: 'Eira', 
               10: 'Crax', 11: 'Procyon', 12: 'Capra', 13: 'Dasypus', 14: 'Sciurus', 
               15: 'Crypturellus', 16: 'Tamandua', 17: 'Proechimys', 18: 'Leopardus', 
               19: 'Equus', 20: 'Columbina', 21: 'Nyctidromus', 22: 'Ortalis', 
               23: 'Emballonura', 24: 'Odontophorus', 25: 'Geotrygon', 26: 'Metachirus', 
               27: 'Catharus', 28: 'Cerdocyon', 29: 'Momotus', 30: 'Tapirus', 31: 'Canis', 
               32: 'Furnarius', 33: 'Didelphis', 34: 'Sylvilagus', 35: 'Unknown'}

# Create necessary folders
os.makedirs(labeled_folder, exist_ok=True)
os.makedirs(species_folder, exist_ok=True)

# Annotate images
for image_data in detections['images']:
    image_path = os.path.join(image_folder, image_data['file'])
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image: {image_path}")
        continue

    highest_prob = 0
    best_bbox = None
    best_label = "Unknown"

    for idx, detection in enumerate(image_data['detections']):
        # Only process animal category
        if detection['category'] == '1':
            bbox = detection['bbox']
            x1, y1, w, h = int(bbox[0] * image.shape[1]), int(bbox[1] * image.shape[0]), \
                           int(bbox[2] * image.shape[1]), int(bbox[3] * image.shape[0])
            x2, y2 = x1 + w, y1 + h

            # Get classification label
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

    # Save annotated image
    labeled_image_path = os.path.join(labeled_folder, os.path.basename(image_data['file']))

    if best_label != "Unknown":
        # Save species-specific images
        species_image_path = os.path.join(species_folder, os.path.basename(image_data['file']))
        cv2.imwrite(species_image_path, image)
        print(f"Saving species image: {species_image_path}")
    else:
        # Save images with unknown species
        cv2.imwrite(labeled_image_path, image)
        print(f"Saving labeled image: {labeled_image_path}")

print("Finished labeling:", labeled_folder)

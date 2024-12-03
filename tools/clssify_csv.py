import os
import json
import torch
from PIL import Image
import torchvision.transforms as transforms
from PytorchWildlife.models import classification as pw_classification

# Paths
json_file = r'E:\result\json\detection\part0output.json'
classification_results_json = r'F:\DATASET\NACTI\classification_results_named.json'

CLASS_NAMES = {
    0: 'Dasyprocta', 1: 'Bos', 2: 'Pecari', 3: 'Mazama', 4: 'Cuniculus',
    5: 'Leptotila', 6: 'Human', 7: 'Aramides', 8: 'Tinamus', 9: 'Eira',
    10: 'Crax', 11: 'Procyon', 12: 'Capra', 13: 'Dasypus', 14: 'Sciurus',
    15: 'Crypturellus', 16: 'Tamandua', 17: 'Proechimys', 18: 'Leopardus',
    19: 'Equus', 20: 'Columbina', 21: 'Nyctidromus', 22: 'Ortalis',
    23: 'Emballonura', 24: 'Odontophorus', 25: 'Geotrygon', 26: 'Metachirus',
    27: 'Catharus', 28: 'Cerdocyon', 29: 'Momotus', 30: 'Tapirus', 31: 'Canis',
    32: 'Furnarius', 33: 'Didelphis', 34: 'Sylvilagus', 35: 'Unknown'
}

# Transform for classification
transform = transforms.Compose([
    transforms.Resize((1280, 1280)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classification_model = pw_classification.AI4GAmazonRainforest(device=device)
classification_model.eval()

classification_results = []

# Load JSON annotations
with open(json_file, 'r') as f:
    annotations = json.load(f)["annotations"]

# Iterate through annotations
for annotation in annotations:
    img_path = annotation["img_id"]
    if not os.path.exists(img_path):
        print(f"Image {img_path} not found, skipping.")
        continue

    # Open image
    image = Image.open(img_path).convert("RGB")
    print(f"Processing image: {img_path}")

    for bbox, category, confidence in zip(annotation["bbox"], annotation["category"], annotation["confidence"]):
        x_min, y_min, x_max, y_max = map(int, bbox)

        # Crop the detected region
        cropped_img = image.crop((x_min, y_min, x_max, y_max))

        # Apply transformations
        input_tensor = transform(cropped_img).unsqueeze(0).to(device)

        # Predict class
        with torch.no_grad():
            logits = classification_model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            max_prob, predicted_class = torch.max(probs, dim=1)

        # Convert predicted class index to name
        predicted_class_name = CLASS_NAMES[predicted_class.item()]

        # Save result
        classification_results.append({
            'img_id': img_path,
            'bbox': bbox,
            'ground_truth_category': CLASS_NAMES[category] if category in CLASS_NAMES else 'Unknown',
            'detection_confidence': confidence,
            'predicted_class': predicted_class_name,
            'classification_confidence': max_prob.item(),
        })
        print(f"Detection {bbox} classified as: {predicted_class_name} with probability: {max_prob.item():.2f}")

# Save classification results to JSON
with open(classification_results_json, 'w', encoding='utf-8') as f:
    json.dump(classification_results, f, indent=4, ensure_ascii=False)
print(f"Classification results saved to {classification_results_json}")

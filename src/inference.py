import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torchvision import transforms
from PytorchWildlife.models import classification as pw_classification
import json
from PIL import Image
MODEL_PATH = "../src/models/focal_loss/best_model.pth"

Class_names = {
    0: 'american black bear', 1: 'american marten', 2: 'american red squirrel', 3: 'black-tailed jackrabbit',
    4: 'bobcat', 5: 'california ground squirrel', 6: 'california quail', 7: 'cougar', 8: 'coyote', 9: 'dark-eyed junco',
    10: 'domestic cow', 11: 'domestic dog', 12: 'donkey', 13: 'dusky grouse', 14: 'eastern gray squirrel',
    15: 'elk', 16: 'ermine', 17: 'european badger', 18: 'gray fox', 19: 'gray jay', 20: 'horse',
    21: 'house wren', 22: 'long-tailed weasel', 23: 'moose', 24: 'mule deer', 25: 'nine-banded armadillo', 26: 'north american porcupine',
    27: 'north american river otter', 28: 'raccoon', 29: 'red deer', 30: 'red fox', 31: 'snowshoe hare',
    32: "steller's jay", 33: 'striped skunk', 34: 'unidentified accipitrid', 35: 'unidentified bird',
    36: 'unidentified chipmunk', 37: 'unidentified corvus', 38: 'unidentified deer', 39: 'unidentified deer mouse',
    40: 'unidentified mouse', 41: 'unidentified pack rat', 42: 'unidentified pocket gopher', 43: 'unidentified rabbit',
    44: 'vehicle', 45: 'virginia opossum', 46: 'wild boar', 47: 'wild turkey', 48: 'yellow-bellied marmot'
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load model
model = pw_classification.AI4GAmazonRainforest(device=device)
num_features = model.net.classifier.in_features
model.net.classifier = torch.nn.Linear(num_features, 49)

checkpoint = torch.load(MODEL_PATH, map_location=device)
state_dict = checkpoint.get("model", checkpoint)
# if ddp is used, the model is saved as 'module.xxx'
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

image_folder = "../test_img"
results = {}

with torch.no_grad():
    for image_name in os.listdir(image_folder):
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            continue

        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).to(device)
        input_tensor = input_tensor.unsqueeze(0)  # add batch dimension

        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_class_idx = predicted.item()
        pred_class_name = Class_names.get(pred_class_idx, "Unknown")

        results[image_name] = {
            "class_index": pred_class_idx,
            "class_name": pred_class_name
        }
        print(f"Processed {image_name}: {pred_class_name}")

output_json_path = "inference_results.json"
with open(output_json_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"Results saved to {output_json_path}")



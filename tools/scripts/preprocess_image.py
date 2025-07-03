import os
import glob
from PIL import Image
import torch
from torchvision import transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

to_pil = transforms.ToPILImage()

input_folder = r"/Users/zehualiu/Desktop/nacti/horse"
output_folder = "../tf/horse_normalized"
os.makedirs(output_folder, exist_ok=True)

image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))
image_paths = image_paths[:5]

for img_path in image_paths:
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img)

    tensor_min = img_tensor.min()
    tensor_max = img_tensor.max()
    img_tensor_vis = (img_tensor - tensor_min) / (tensor_max - tensor_min)

    img_pil = to_pil(img_tensor_vis)

    file_name = os.path.splitext(os.path.basename(img_path))[0] + "_normalised.jpg"
    save_path = os.path.join(output_folder, file_name)
    img_pil.save(save_path)
    print(f"Saving: {save_path}")
import os
from PIL import Image
from src.dataset import NACTIAnnotationDataset

output_dir = r"E:\result\image"
os.makedirs(output_dir, exist_ok=True)

dataset = NACTIAnnotationDataset(
    image_dir=r"F:\DATASET\NACTI\images",
    json_path=r"E:\result\json\detection\detection_filtered.json",
    csv_path=r"F:/DATASET/NACTI/meta/nacti_metadata_balanced.csv"
)

selected_images = {}

for idx in range(len(dataset)):
    image, target = dataset[idx]
    common_name = target["common_name"]
    if common_name not in selected_images:
        selected_images[common_name] = image
        print(f"add：{common_name}")

for common_name, image in selected_images.items():
    filename = f"{common_name}.jpg"
    output_path = os.path.join(output_dir, filename)
    image.save(output_path)
    print(f"已保存：{output_path}")
import os
import json
from PIL import Image

# paths
image_folder = r'G:\result\detected'
json_file = 'part3output.json'
output_folder = r'G:\result\cropped_images'

# mkdir
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# decode JSON file and crop detected objects
with open(json_file) as f:
    detections = json.load(f)

for image_data in detections['images']:
    image_path = os.path.join(image_folder, image_data['file'])
    image = Image.open(image_path)

    for idx, detection in enumerate(image_data['detections']):
        if detection['category'] == '1':  # only process animal category
            bbox = detection['bbox']
            x, y, w, h = bbox
            cropped_image = image.crop((x * image.width, y * image.height,
                                        (x + w) * image.width, (y + h) * image.height))
            # save cropped image
            cropped_image_name = f"{os.path.splitext(os.path.basename(image_data['file']))[0]}_crop_{idx}.png"
            cropped_image_path = os.path.join(output_folder, cropped_image_name)
            cropped_image.save(cropped_image_path)
            print(f"saving cropped image: {cropped_image_path}")


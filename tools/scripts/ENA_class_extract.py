import os
import json
import shutil

json_file = r'F:\DATASET\ENA24-Detection\metadata\ena24_updated.json'
images_dir = r'F:\DATASET\ENA24-Detection\images'
wildturkey_dir = r'E:\result\ena_image\wildturkey'
horse_dir = r'E:\result\ena_image\horse'

os.makedirs(wildturkey_dir, exist_ok=True)
os.makedirs(horse_dir, exist_ok=True)

with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

image_dict = {img['id']: img['file_name'] for img in data['images']}

wildturkey_cat_id = None
horse_cat_id = None
for cat in data['categories']:
    if cat['name'].lower() == 'wild turkey':
        wildturkey_cat_id = cat['id']
    elif cat['name'].lower() == 'horse':
        horse_cat_id = cat['id']

if wildturkey_cat_id is None or horse_cat_id is None:
    print("Category ID not found for Wild Turkey or Horse.")
    exit(1)

wildturkey_image_ids = set()
horse_image_ids = set()

for ann in data['annotations']:
    if ann['category_id'] == wildturkey_cat_id:
        wildturkey_image_ids.add(ann['image_id'])
    if ann['category_id'] == horse_cat_id:
        horse_image_ids.add(ann['image_id'])

for img_id in wildturkey_image_ids:
    file_name = image_dict.get(img_id)
    if file_name:
        src_path = os.path.join(images_dir, file_name)
        dst_path = os.path.join(wildturkey_dir, file_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            print(f"copy {src_path} to {dst_path}")
        else:
            print(f"file {src_path} cannot find.")

for img_id in horse_image_ids:
    file_name = image_dict.get(img_id)
    if file_name:
        src_path = os.path.join(images_dir, file_name)
        dst_path = os.path.join(horse_dir, file_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            print(f"copy {src_path} to {dst_path}")
        else:
            print(f"file {src_path} cannot find")

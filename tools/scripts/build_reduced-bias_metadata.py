import json

Class_names = {
    0: 'american black bear', 1: 'american marten', 2: 'american red squirrel', 3: 'black-tailed jackrabbit',
    4: 'bobcat', 5: 'california ground squirrel', 6: 'california quail', 7: 'cougar', 8: 'coyote', 9: 'dark-eyed junco',
    10: 'domestic cow', 11: 'domestic dog', 12: 'donkey', 13: 'dusky grouse', 14: 'eastern gray squirrel',
    15: 'elk', 16: 'ermine', 17: 'european badger', 18: 'gray fox', 19: 'gray jay', 20: 'horse',
    21: 'house wren', 22: 'long-tailed weasel', 23: 'moose', 24: 'mule deer', 25: 'nine-banded armadillo',
    26: 'north american porcupine', 27: 'north american river otter', 28: 'raccoon', 29: 'red deer', 30: 'red fox',
    31: 'snowshoe hare', 32: "steller's jay", 33: 'striped skunk', 34: 'unidentified accipitrid',
    35: 'unidentified bird', 36: 'unidentified chipmunk', 37: 'unidentified corvus', 38: 'unidentified deer',
    39: 'unidentified deer mouse', 40: 'unidentified mouse', 41: 'unidentified pack rat',
    42: 'unidentified pocket gopher', 43: 'unidentified rabbit', 44: 'vehicle', 45: 'virginia opossum',
    46: 'wild boar', 47: 'wild turkey', 48: 'yellow-bellied marmot'
}

name_to_new_id = {v: k for k, v in Class_names.items()}

def filter_dataset(input_json_path, output_json_path):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_annotations = data.get('annotations', [])
    original_categories = data.get('categories', [])
    original_images = data.get('images', [])

    old_id_to_new_id = {}
    filtered_categories = []

    for cat in original_categories:
        raw_cat_name = cat['name']
        old_id = cat['id']
        normalized_cat_name = raw_cat_name.replace('_', ' ').strip().lower()

        if normalized_cat_name in name_to_new_id:
            new_id = name_to_new_id[normalized_cat_name]
            old_id_to_new_id[old_id] = new_id

            filtered_categories.append({
                "id": new_id,
                # save blank as NACTI metadata format
                "name": normalized_cat_name
            })

    print(f"Overlapped {len(filtered_categories)} samples")

    filtered_annotations = []
    kept_image_ids = set()

    for ann in original_annotations:
        old_cat_id = ann.get('category_id')

        if old_cat_id in old_id_to_new_id:
            new_ann = ann.copy()
            new_ann['category_id'] = old_id_to_new_id[old_cat_id]
            filtered_annotations.append(new_ann)

            if 'image_id' in ann:
                kept_image_ids.add(ann['image_id'])
            elif 'id' in ann:
                kept_image_ids.add(ann['id'])

    filtered_images = []
    for img in original_images:
        if img.get('id') in kept_image_ids:
            filtered_images.append(img)

    print(f"Original sample: {len(original_annotations)} -> filtered samples: {len(filtered_annotations)}")

    if len(filtered_annotations) != 0:
        new_data = {
            "images": filtered_images,
            "annotations": filtered_annotations,
            "categories": filtered_categories
        }

        if 'info' in data:
            new_data['info'] = data['info']
        if 'licenses' in data:
            new_data['licenses'] = data['licenses']

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=4)

        print(f"\n saving new json file to: {output_json_path}")
    else:
        print("No valid annotations found after filtering. No output file created.")

if __name__ == "__main__":
    filter_dataset(r"H:\Downloads\Download\missouri_camera_traps_set1.json", r"H:\Downloads\Download\missouri_camera_traps_set1_updated.json")

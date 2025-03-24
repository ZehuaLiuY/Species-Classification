import json

class_names = {
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

def update_and_filter_json(json_path, output_path, class_names):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_categories = []
    # old category_id -> new category_id
    old_to_new = {}
    for cat in data.get("categories", []):
        orig_id = cat["id"]
        cat_name = cat["name"].lower().replace('_', ' ')
        new_id = None
        for key, name in class_names.items():
            if name == cat_name:
                new_id = key
                break
        if new_id is not None:
            print(f"Updating '{cat['name']}' id from {orig_id} to {new_id}")
            old_to_new[orig_id] = new_id
            cat["id"] = new_id
            new_categories.append(cat)
        else:
            print(f"delete class '{cat['name']}', because not in class_names")
    data["categories"] = new_categories

    # process annotations: update category_id
    new_annotations = []
    for ann in data.get("annotations", []):
        old_cat_id = ann.get("category_id")
        if old_cat_id in old_to_new:
            new_cat_id = old_to_new[old_cat_id]
            ann["category_id"] = new_cat_id
            new_annotations.append(ann)
            print(f"updating annotation id {ann.get('id')} category_id from {old_cat_id} to {new_cat_id}")
        else:
            print(f"deleting annotation id {ann.get('id')} category_id {old_cat_id}, because category_id not in old_to_new")
    data["annotations"] = new_annotations

    # save updated json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"updated json save to '{output_path}'")

if __name__ == '__main__':

    input_json = r"F:/DATASET/ENA24-Detection/metadata/ena24.json"
    output_json = r"F:/DATASET/ENA24-Detection/metadata/ena24_updated.json"
    update_and_filter_json(input_json, output_json, class_names)
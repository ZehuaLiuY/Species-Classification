import json

# part0, 1, 2, 3 are the json files that need to be merged
json_files = []

merged_annotations = []

for file in json_files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if "annotations" in data:
            merged_annotations.extend(data["annotations"])

merged_data = {
    "annotations": merged_annotations
}

with open("merged_detection.json", "w", encoding='utf-8') as f:
    json.dump(merged_data, f, indent=4, ensure_ascii=False)

print("JSON saved as merged_detection.json")
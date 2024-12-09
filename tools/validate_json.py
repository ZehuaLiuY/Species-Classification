import json

def is_valid_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        print("JSON file is valid.")
        return True
    except json.JSONDecodeError as e:
        print(f"JSON is not valid: {e}")
        return False
    except Exception as e:
        print(f"cannot read file: {e}")
        return False

file_path = r'F:\DATASET\NACTI\processed_meta\classification_part0_formatted.json'
is_valid_json(file_path)

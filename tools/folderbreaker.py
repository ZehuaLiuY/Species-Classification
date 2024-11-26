import os
import shutil

source_path = r"F:\DATASET\NACTI\images\part0"

target_folder = r"F:\DATASET\NACTI\images\nacti_part0"

os.makedirs(target_folder, exist_ok=True)

for root, dirs, files in os.walk(source_path):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(root, file)
            try:
                shutil.copy(file_path, os.path.join(target_folder, file))
            except Exception as e:
                print(f"Error copying {file_path}: {e}")

print(f"All images are now in: {target_folder}")

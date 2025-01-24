import os
import shutil

def move_files_to_root(directory):
    subfolders = [os.path.join(directory, subfolder) for subfolder in os.listdir(directory) if os.path.isdir(os.path.join(directory, subfolder))]

    for subfolder in subfolders:
        for filename in os.listdir(subfolder):
            file_path = os.path.join(subfolder, filename)
            if os.path.isfile(file_path):
                shutil.move(file_path, directory)

        os.rmdir(subfolder)

images_directory = "images"

move_files_to_root(images_directory)

print("Files moved to root directory.")

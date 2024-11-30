import os
import torch
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife import utils as pw_utils
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

detection_model = pw_detection.MegaDetectorV6(device=device, pretrained=True, version="yolov9c")
dataset_path = r"F:\DATASET\NACTI\images\nacti_part0"

results = detection_model.batch_image_detection(dataset_path, batch_size=128)
batch_output = r'E:\result\annotated_images'
crop_output = r'E:\result\crops'
print("Saving results...")
pw_utils.save_detection_json(results, os.path.join(".","part0output.json"),
                             categories=detection_model.CLASS_NAMES,
                             exclude_category_ids=[], # Category IDs can be found in the definition of each model.
                             exclude_file_path=None)
# Save images
pw_utils.save_detection_images(results, "batch_output", overwrite=False)
pw_utils.save_crop_images(results, "crop_output", overwrite=False)

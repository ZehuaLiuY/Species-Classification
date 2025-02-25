import pandas as pd
from torch.utils.data import Dataset
from dataset import NACTIAnnotationDataset


class MultimodalNACTIDataset(Dataset):
    def __init__(self, image_dir, json_path, csv_path, text_csv_path=None, text_mapping=None, transforms=None, allow_empty=False):
        """
        :param text_csv_path: file for common_name to description mapping
        :param text_mapping: dict for common_name to description mapping
        """
        # init base dataset
        self.base_dataset = NACTIAnnotationDataset(image_dir, json_path, csv_path, transforms=transforms, allow_empty=allow_empty)

        # load text mapping
        if text_mapping is not None:
            self.text_mapping = text_mapping
        elif text_csv_path is not None:
            df = pd.read_csv(text_csv_path)
            # key: common_name, value: description
            self.text_mapping = dict(zip(df['common_name'], df['description']))
        else:
            # if no text mapping provided, use common_name as text
            self.text_mapping = {}

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # get image, text and target from base dataset
        image, target = self.base_dataset[idx]
        # get text from common_name
        common_name = target.get("common_name", "")
        text = self.text_mapping.get(common_name, common_name)
        return image, text, target

# test
if __name__ == "__main__":
    image_dir = r"F:\DATASET\NACTI\images"
    json_path = r"E:\result\json\detection\detection_filtered.json"
    csv_path = r"F:/DATASET/NACTI/meta/nacti_metadata_balanced.csv"
    text_csv_path = r"F:/DATASET/NACTI/meta/common_name_desc.csv"

    # optional: text_mapping
    # e.g.
    # text_mapping = {"red deer": "A large deer species commonly found...", ...}

    multimodal_dataset = MultimodalNACTIDataset(
        image_dir=image_dir,
        json_path=json_path,
        csv_path=csv_path,
        text_csv_path=text_csv_path,
        transforms=None,
        allow_empty=False
    )

    image, text, target = multimodal_dataset[0]
    print("Image size:", image.size)
    print("Text:", text)
    print("Target:", target)
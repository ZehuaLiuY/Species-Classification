import os

import pandas as pd

pred_path = r'E:\result\json\classification\classification_part0.json'
gt_path = r'F:\DATASET\NACTI\meta\nacti_metadata_part0.csv'

pred_df = pd.read_json(pred_path)
gt_df = pd.read_csv(gt_path)

print(gt_df.columns)

gt_df['filename'] = gt_df['filename'].apply(lambda x: x.split("\\")[-1])
pred_df['img_id'] = pred_df['img_id'].apply(lambda x: x.split("/")[-1])

gt_df['filename'] = gt_df['filename'].str.replace(r'^.*part0/sub\d+/', '', regex=True)
pred_df['img_id'] = pred_df['img_id'].str.replace(r'^.*\\', '', regex=True)

# save the processed data
output_dir = r'F:\DATASET\NACTI\processed_meta'
os.makedirs(output_dir, exist_ok=True)
gt_df.to_csv(os.path.join(output_dir, 'nacti_metadata_part0.csv'), index=False)
pred_output_path = os.path.join(output_dir, 'classification_part0.json')
pred_df.to_json(pred_output_path, orient='records', indent=4, force_ascii=False)

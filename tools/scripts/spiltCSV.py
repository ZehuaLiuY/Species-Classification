import pandas as pd
import os

input_file = r'H:\DATASET\NACTI\meta\nacti_metadata.csv'
output_path = r'H:\DATASET\NACTI\meta'

target_columns = [
    'seq_no', 'id', 'filename', 'study', 'location', 'width', 'height',
    'category_id', 'name', 'genus', 'family', 'order', 'class', 'common_name'
]
df = pd.read_csv(input_file, low_memory=False)
part3_df = df[df['filename'].str.contains('part0')]
filtered_df = part3_df[target_columns]
filtered_df.to_csv(os.path.join(output_path, "nacti_metadata_part0.csv"), index=False)


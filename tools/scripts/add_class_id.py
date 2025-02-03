import pandas as pd

species_list = [
    'american black bear', 'american marten', 'american red squirrel', 'black-tailed jackrabbit',
    'bobcat', 'california ground squirrel', 'california quail', 'cougar', 'coyote',
    'dark-eyed junco', 'domestic cow', 'domestic dog', 'donkey', 'dusky grouse',
    'eastern gray squirrel', 'elk', 'empty', 'ermine', 'european badger', 'gray fox',
    'gray jay', 'horse', 'house wren', 'long-tailed weasel', 'moose', 'mule deer',
    'nine-banded armadillo', 'north american porcupine', 'north american river otter',
    'raccoon', 'red deer', 'red fox', 'snowshoe hare', "steller's jay", 'striped skunk',
    'unidentified accipitrid', 'unidentified bird', 'unidentified chipmunk',
    'unidentified corvus', 'unidentified deer', 'unidentified deer mouse',
    'unidentified mouse', 'unidentified pack rat', 'unidentified pocket gopher',
    'unidentified rabbit', 'vehicle', 'virginia opossum', 'wild boar', 'wild turkey',
    'yellow-bellied marmot'
]

df = pd.read_csv("F:/DATASET/NACTI/meta/nacti_metadata_balanced.csv", low_memory=False)

if 'common_name' in df.columns:
    df['class_ID'] = df['common_name'].map(lambda x: species_list.index(x) if x in species_list else -1)

    # 保存新的 CSV 文件
    df.to_csv("F:/DATASET/NACTI/meta/nacti_metadata_balanced_class.csv", index=False)
    print("new csv saved with `class_ID`！")
else:
    print("Error: No `common_name` column in the CSV file.")

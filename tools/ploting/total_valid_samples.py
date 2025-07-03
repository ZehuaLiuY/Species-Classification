import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

datapath = r'/Users/zehualiu/Documents/GitHub/Species-Classification/metadata/nacti_metadata_balanced.csv'
df = pd.read_csv(datapath, low_memory=False)

df = df[~df['common_name'].isin(['empty'])]

json_path = r'/Users/zehualiu/Documents/GitHub/Species-Classification/metadata/detection_filtered.json'
with open(json_path, 'r') as f:
    data = json.load(f)

json_ids = {
    os.path.basename(ann['img_id'].replace('\\','/'))
    for ann in data['annotations']
    if len(ann.get('bbox', [])) > 0
}


df = df[df['id'].isin(json_ids)]
species_counts = df['common_name'].value_counts()
cum_pct = species_counts.cumsum() / species_counts.sum()

head_cutoff_species = cum_pct[cum_pct >= 0.5].index[0]
head_cutoff_position = species_counts.index.get_loc(head_cutoff_species)

colors = []
for i, count in enumerate(species_counts.values):
    if count <= 20:
        colors.append("green")
    else:
        colors.append("lightblue" if i <= head_cutoff_position else "pink")

plt.figure(figsize=(45, 20))
bars = plt.bar(species_counts.index, species_counts.values, color=colors)
plt.yscale("log")

for bar, value in zip(bars, species_counts.values):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        value,
        f"{value}",
        ha='center',
        va='bottom',
        fontsize=32,
        rotation=45
    )


num_classes = species_counts.shape[0]
total_instances = species_counts.sum()

print(f"Number of classes: {num_classes}")
print(f"Total instances across all classes: {total_instances}\n")

print("Class".ljust(30), "Count")
print("-"*40)
for species, count in species_counts.items():
    print(f"{species.ljust(30)} {count}")

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.ylabel('Count (log scale)', fontsize=32, fontweight='bold')
plt.xlabel('Species', fontsize=27, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=32, fontweight='bold')
plt.yticks(fontsize=32, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

head_patch = mpatches.Patch(color='lightblue', label='Head (≤50% cumulative)')
tail_patch = mpatches.Patch(color='pink',      label='Tail (>50%)')
few_patch  = mpatches.Patch(color='green',     label='Few-shot (≤20)')
plt.legend(handles=[head_patch, tail_patch, few_patch],
           fontsize=32, handleheight=3, handlelength=4)

plt.show()
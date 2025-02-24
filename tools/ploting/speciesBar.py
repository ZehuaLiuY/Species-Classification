import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

datapath = r'/Users/zehualiu/Documents/GitHub/Project-Prep/metadata/nacti_metadata_balanced.csv'
datapath = r'/Users/zehualiu/Documents/GitHub/Project-Prep/metadata/nacti_metadata.csv'
df = pd.read_csv(datapath, low_memory=False)
species_counts = df['common_name'].value_counts()

cum_pct = species_counts.cumsum() / species_counts.sum()

head_cutoff_species = cum_pct[cum_pct >= 0.5].index[0]
head_cutoff_position = species_counts.index.get_loc(head_cutoff_species)

colors = []
for i, (species, count) in enumerate(species_counts.items()):

    if count <= 20:
        colors.append("green")
    else:

        if i <= head_cutoff_position:
            colors.append("lightblue")
        else:
            colors.append("pink")

plt.figure(figsize=(50, 25))
bars = plt.bar(species_counts.index, species_counts.values, color=colors)
plt.yscale("log")

for bar, value in zip(bars, species_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             value,
             f"{value}",
             ha='center',
             va='bottom',
             fontsize=16,
             rotation=45)

plt.ylabel('Count (log scale)', fontsize=16)
plt.xlabel('Species', fontsize=16)
plt.xticks(rotation=90, ha='right', fontsize=16)
plt.yticks(fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

head_patch = mpatches.Patch(color='lightblue', label='Head (≤50% cumulative)')
tail_patch = mpatches.Patch(color='pink',      label='Tail (>50%)')
few_patch  = mpatches.Patch(color='green',     label='Few-shot (≤20)')
plt.legend(handles=[head_patch, tail_patch, few_patch], fontsize=16)

plt.savefig('species_count_vertical.pdf', dpi=1200)
plt.show()
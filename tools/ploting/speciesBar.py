import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

datapath = r'/Users/zehualiu/Documents/GitHub/Species-Classification/metadata/nacti_metadata_balanced.csv'
# datapath = r'/Users/zehualiu/Documents/GitHub/Project-Prep/metadata/nacti_metadata.csv'
df = pd.read_csv(datapath, low_memory=False)
df = df[df['common_name'] != 'vehicle']
df = df[df['common_name'] != 'empty']
species_counts = df['common_name'].value_counts()

cum_pct = species_counts.cumsum() / species_counts.sum()

head_cutoff_species = cum_pct[cum_pct >= 0.5].index[0]
head_cutoff_position = species_counts.index.get_loc(head_cutoff_species)

colors = []
for i, (species, count) in enumerate(species_counts.items()):
    print(f"{species}: {count}")

    if count <= 20:
        colors.append("green")
    else:

        if i <= head_cutoff_position:
            colors.append("lightblue")
        else:
            colors.append("pink")

plt.figure(figsize=(45, 20))
bars = plt.bar(species_counts.index, species_counts.values, color=colors)
plt.yscale("log")

for bar, value in zip(bars, species_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             value,
             f"{value}",
             ha='center',
             va='bottom',
             fontsize=32,
             rotation=45)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.ylabel('Count (log scale)', fontsize=32, fontweight='bold')
plt.xlabel('Species', fontsize=27,fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=32, fontweight = 'bold')
plt.yticks(fontsize=32, fontweight = 'bold')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

head_patch = mpatches.Patch(color='lightblue', label='Head (≤50% cumulative)')
tail_patch = mpatches.Patch(color='pink',      label='Tail (>50%)')
few_patch  = mpatches.Patch(color='green',     label='Few-shot (≤20)')
plt.legend(handles=[head_patch, tail_patch, few_patch], fontsize=32, handleheight=3, handlelength=4)

plt.savefig('species_count_vertical.pdf', dpi=2400)
# plt.show()
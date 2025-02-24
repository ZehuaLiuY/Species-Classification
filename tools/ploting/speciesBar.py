import pandas as pd
import matplotlib.pyplot as plt

datapath = r'/Users/zehualiu/Documents/GitHub/Project-Prep/metadata/nacti_metadata.csv'
df = pd.read_csv(datapath)

species_counts = df['common_name'].value_counts()

plt.figure(figsize=(50, 25))

cmap = plt.get_cmap("coolwarm")
colors = cmap(species_counts.rank(pct=True).values)

bars = plt.bar(species_counts.index, species_counts.values, color=colors)

plt.yscale("log")

for bar, value in zip(bars, species_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, value, f"{value}",
             ha='center', va='bottom', fontsize=16, color="black", rotation=45)

plt.ylabel('Count (log scale)', fontsize=16)
plt.xlabel('Species', fontsize=16)
# plt.title(f'Species Count in NACTI Dataset', fontsize=14)
plt.xticks(rotation=90, ha='right',fontsize=16)
plt.yticks(fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('species_count_vertical.pdf', dpi=1200)
plt.show()
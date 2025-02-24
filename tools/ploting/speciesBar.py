import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

datapath = r'/Users/zehualiu/Documents/GitHub/Project-Prep/metadata/nacti_metadata.csv'
df = pd.read_csv(datapath, low_memory=False)
species_counts = df['common_name'].value_counts()  # 默认已按样本数降序排序

# 计算累计比例（当前类别包含在内）
cum_pct = species_counts.cumsum() / species_counts.sum()

cmap_custom = mcolors.LinearSegmentedColormap.from_list('custom', ['lightblue', 'red'])
min_norm = 0.3

colors = []
for i, (species, count) in enumerate(species_counts.items()):
    if count <= 20:
        colors.append('navy')
    # 如果当前类别加入后累计比例还未超过50%，或者刚好超过50%，都标红
    elif cum_pct.iloc[i] < 0.5 or (i > 0 and cum_pct.iloc[i-1] < 0.5 and cum_pct.iloc[i] >= 0.5):
        colors.append(cmap_custom(1))  # 红色
    else:
        # 对于累计比例超过50%的类别，计算从红色到浅蓝色的渐变
        # 这里将累积比例从0.5-1.0映射到梯度0-1（注意：cmap_custom中0对应lightblue，1对应red，所以需要反转）
        fraction = (cum_pct.iloc[i] - 0.5) / 0.5  # 0到1之间
        norm = 1 - fraction * (1 - min_norm)
        colors.append(cmap_custom(norm))

plt.figure(figsize=(50, 25))
bars = plt.bar(species_counts.index, species_counts.values, color=colors)
plt.yscale("log")

for bar, value in zip(bars, species_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, value, f"{value}",
             ha='center', va='bottom', fontsize=16, color="black", rotation=45)

plt.ylabel('Count (log scale)', fontsize=16)
plt.xlabel('Species', fontsize=16)
plt.xticks(rotation=90, ha='right', fontsize=16)
plt.yticks(fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

norm_for_cbar = mcolors.Normalize(vmin=min_norm, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap_custom, norm=norm_for_cbar)
sm.set_array([])
cbar = plt.colorbar(sm, pad=0.02)
cbar.set_label('Transition Mapping (for count >20)', fontsize=16)
cbar.set_ticks([min_norm, (min_norm+1)/2, 1])
cbar.set_ticklabels([f'≈LightBlue', 'Transition', 'Red (Cumulative 50%)'])

low_count_patch = mpatches.Patch(color='navy', label='Count ≤20')
plt.legend(handles=[low_count_patch], loc='upper right', fontsize=16)

plt.savefig('species_count_vertical.pdf', dpi=1200)
plt.show()
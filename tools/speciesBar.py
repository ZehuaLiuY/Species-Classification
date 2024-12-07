import pandas as pd
import matplotlib.pyplot as plt

datapath = r'F:\DATASET\NACTI\meta\nacti_metadata.csv'
df = pd.read_csv(datapath)

species_counts = df['common_name'].value_counts()
print(species_counts)

plt.figure(figsize=(10, 20))
ax = species_counts.plot(kind='barh')
plt.xlabel('Count')
plt.ylabel('Species')
plt.title('Species Count in Dataset')
plt.tight_layout()

for index, value in enumerate(species_counts):
    ax.text(value, index, str(value), va='center', fontsize=8, color='black')

plt.show()

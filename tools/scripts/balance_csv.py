import pandas as pd

datapath = r'/Users/zehualiu/Documents/GitHub/Project-Prep/metadata/nacti_metadata_part0.csv'
metadata = pd.read_csv(datapath)

# counter the number of samples for each species
species_counts = metadata['common_name'].value_counts()

# create a list to store balanced data
balanced_data = []

for species, count in species_counts.items():
    species_data = metadata[metadata['common_name'] == species]
    if count > 100000:
        species_data = species_data.sample(n=100000, random_state=42)

    balanced_data.append(species_data)

balanced_data = pd.concat(balanced_data)

balanced_data.to_csv("balanced_metadata.csv", index=False)

print("new csv filed saved as balanced_metadata_part0.csv")
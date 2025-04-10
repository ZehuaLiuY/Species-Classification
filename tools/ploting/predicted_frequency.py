import json
import matplotlib.pyplot as plt
from collections import Counter
json_path = r'G:\Code\github\Project-Prep\test_result\nonbiased\json\48\LDAM.json'
with open(json_path, 'r') as f:
    data = json.load(f)

filtered_items = [entry for entry in data if entry.get("ground_truth_class") in ["horse"]]

counter = Counter(entry.get("predicted_class") for entry in filtered_items)

plt.figure(figsize=(15, 6))
plt.bar(list(counter.keys()), list(counter.values()))
plt.xlabel('Predicted Class')
plt.ylabel('Frequency')
plt.title('Predicted Frequency for Entries with Ground Truth "horse"')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('predicted_frequency_horse.pdf', dpi=600)
plt.show()

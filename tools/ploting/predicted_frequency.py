import json
import matplotlib.pyplot as plt
from collections import Counter
json_path = r'G:\Code\github\Project-Prep\test_result\nonbiased\json\48\LDAM.json'
with open(json_path, 'r') as f:
    data = json.load(f)

filtered_items = [entry for entry in data if entry.get("ground_truth_class") == "wild turkey"]
counter = Counter(entry["predicted_class"] for entry in filtered_items)

sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
classes, freqs = zip(*sorted_items)

plt.figure(figsize=(15, 6))
plt.bar(classes, freqs)
plt.xlabel('Predicted Class')
plt.ylabel('Frequency')
plt.title('Predicted Frequency for Entries with Ground Truth "wild turkey"')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('predicted_frequency_horse.pdf', dpi=600)
plt.show()

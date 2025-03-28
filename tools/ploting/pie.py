import json
import matplotlib.pyplot as plt

json_path = r"F:\DATASET\ENA24-Detection\metadata\ena24_updated.json"

with open(json_path, "r") as f:
    data = json.load(f)

annotation_counts = {}
for ann in data["annotations"]:
    cat_id = ann["category_id"]
    annotation_counts[cat_id] = annotation_counts.get(cat_id, 0) + 1

cat_names = {cat["id"]: cat["name"] for cat in data["categories"]}


labels = []
sizes = []
for cat_id, count in annotation_counts.items():
    labels.append(cat_names.get(cat_id, str(cat_id)))
    sizes.append(count)

plt.figure(figsize=(9,9))
plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
plt.title("Unbiased Test Set Species Distribution")
plt.axis("equal")
plt.savefig("Unbiased Distribution.pdf", dpi=600)
plt.show()

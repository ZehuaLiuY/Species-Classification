import json
import pandas as pd
import matplotlib.pyplot as plt

json_path = r'G:\Code\github\Project-Prep\test_result\nonbiased\json\48\CE_Adam.json'
with open(json_path, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# last word of a string
def get_last_word(s):
    return s.split()[-1].lower()

df["correct"] = df.apply(lambda row: get_last_word(row["predicted_class"]) == get_last_word(row["ground_truth_class"]), axis=1)
df["gt_last_word"] = df["ground_truth_class"].apply(get_last_word)

grouped = df.groupby("gt_last_word")["correct"].agg(["sum", "count"])
grouped["accuracy"] = grouped["sum"] / grouped["count"]

plt.figure(figsize=(10, 6))
plt.bar(grouped.index, grouped["accuracy"])
plt.xlabel("Species")
plt.ylabel("Accuracy")
plt.title("Per-Class Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
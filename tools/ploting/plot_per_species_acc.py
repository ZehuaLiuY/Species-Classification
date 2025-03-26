import json
import pandas as pd
import matplotlib.pyplot as plt

json_path = r'G:\Code\github\Project-Prep\test_result\nonbiased\json\48\CE_Adam.json'
with open(json_path, "r") as f:
    data = json.load(f)
df = pd.DataFrame(data)

def get_last_word(s):
    return s.split()[-1].lower()

# df["correct"] = df.apply(lambda row: get_last_word(row["predicted_class"]) == get_last_word(row["ground_truth_class"]), axis=1)
# df["gt_last_word"] = df["ground_truth_class"].apply(get_last_word)
#
# grouped = df.groupby("gt_last_word")["correct"].agg(["sum", "count"])
# grouped["accuracy"] = grouped["sum"] / grouped["count"]

df["correct"] = df["predicted_class"] == df["ground_truth_class"]

grouped = df.groupby("ground_truth_class")["correct"].agg(["sum", "count"])
grouped["accuracy"] = grouped["sum"] / grouped["count"]

grouped = grouped.sort_values(by="count", ascending=False)

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.bar(grouped.index, grouped["accuracy"], color='skyblue', edgecolor='black')
ax1.set_xlabel("Species")
ax1.set_ylabel("Accuracy")
ax1.set_title("Per-Class Accuracy with Sample Count")
ax1.set_ylim(0, 1.0)
plt.xticks(rotation=45)

ax2 = ax1.twinx()
ax2.plot(grouped.index, grouped["count"], color='black', linestyle='-')
ax2.set_ylabel("Sample Count")

plt.tight_layout()
plt.show()
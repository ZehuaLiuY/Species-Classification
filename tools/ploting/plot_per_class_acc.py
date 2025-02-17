import json
import pandas as pd
import matplotlib.pyplot as plt

macro_path = r'/Users/zehualiu/Documents/GitHub/Project-Prep/test_result/46/results_macro.json'
micro_path = r'/Users/zehualiu/Documents/GitHub/Project-Prep/test_result/46/results_micro.json'
weighted_path = r'/Users/zehualiu/Documents/GitHub/Project-Prep/test_result/46/results_weighted.json'

focal_loss_path = r'/Users/zehualiu/Documents/GitHub/Project-Prep/test_result/forcal_loss_49.json'
weighted_cross_path = r'/Users/zehualiu/Documents/GitHub/Project-Prep/test_result/weightedCrossEntropy_49.json'
cross_path = r'/Users/zehualiu/Documents/GitHub/Project-Prep/test_result/results_weighted.json'

def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

df_macro = load_json(focal_loss_path)
df_micro = load_json(weighted_cross_path)
df_weighted = load_json(cross_path)

def compute_accuracy(df):
    df["correct"] = df["predicted_class"] == df["ground_truth_class"]
    print(f"correct predicted number: {df['correct'].sum()} / {len(df)}")
    return df.groupby("ground_truth_class")["correct"].mean()

print("Focal Loss:")
accuracy_macro = compute_accuracy(df_macro)
print("Weight Balanced Cross Entropy:")
accuracy_micro = compute_accuracy(df_micro)
print("Cross Entropy:")
accuracy_weighted = compute_accuracy(df_weighted)


class_prevalence = df_weighted['ground_truth_class'].value_counts()
common_order = class_prevalence.index

def plot_accuracy(accuracy, title, order):
    accuracy_ordered = accuracy.reindex(order)
    plt.figure(figsize=(10, 6))
    accuracy_ordered.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

plot_accuracy(accuracy_macro, "Focal Loss Accuracy per Class", common_order)
plot_accuracy(accuracy_micro, "Weighted Cross Entropy Accuracy per Class", common_order)
plot_accuracy(accuracy_weighted, "Cross Entropy (Baseline) Accuracy per Class", common_order)
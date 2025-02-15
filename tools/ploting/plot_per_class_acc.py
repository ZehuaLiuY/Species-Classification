import json
import pandas as pd
import matplotlib.pyplot as plt


macro_path = r'/Users/zehualiu/Documents/GitHub/Project-Prep/test_result/results_macro.json'
micro_path = r'/Users/zehualiu/Documents/GitHub/Project-Prep/test_result/results_micro.json'
weighted_path = r'/Users/zehualiu/Documents/GitHub/Project-Prep/test_result/results_weighted.json'

def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

df_macro = load_json(macro_path)
df_micro = load_json(micro_path)
df_weighted = load_json(weighted_path)

def compute_accuracy(df):
    df["correct"] = df["predicted_class"] == df["ground_truth_class"]
    # print total corrected predicted class
    print(f"correct predicted number: {df['correct'].sum()} / {len(df)}")
    return df.groupby("ground_truth_class")["correct"].mean()

accuracy_macro = compute_accuracy(df_macro)
accuracy_micro = compute_accuracy(df_micro)
accuracy_weighted = compute_accuracy(df_weighted)

def plot_accuracy(accuracy, title):
    plt.figure(figsize=(10, 6))
    accuracy.sort_values().plot(kind="bar", color="skyblue", edgecolor="black")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()

plot_accuracy(accuracy_macro, "Macro Accuracy per Class")
plot_accuracy(accuracy_micro, "Micro Accuracy per Class")
plot_accuracy(accuracy_weighted, "Weighted Accuracy per Class")

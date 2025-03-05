import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, FuncFormatter

# File paths for JSON results
macro_path = r'/Users/zehualiu/Documents/GitHub/Species-Classification/test_result/46/results_macro.json'
micro_path = r'/Users/zehualiu/Documents/GitHub/Species-Classification/test_result/46/results_micro.json'
weighted_path = r'/Users/zehualiu/Documents/GitHub/Species-Classification/test_result/46/results_weighted.json'

focal_loss_path = r'/Users/zehualiu/Documents/GitHub/Species-Classification/test_result/forcal_loss_49.json'
weighted_cross_path = r'/Users/zehualiu/Documents/GitHub/Species-Classification/test_result/weightedCrossEntropy_49.json'
cross_path = r'/Users/zehualiu/Documents/GitHub/Species-Classification/test_result/results_weighted.json'

# Function to load JSON data into a Pandas DataFrame
def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

# Load data for different loss functions
df_macro = load_json(focal_loss_path)      # Data for Focal Loss
df_micro = load_json(weighted_cross_path)    # Data for Weighted Cross Entropy
df_weighted = load_json(cross_path)          # Data for Cross Entropy (Baseline)

# Function to compute per-class accuracy
def compute_accuracy(df):
    # Create a boolean column 'correct' indicating if the predicted class equals the ground truth
    df["correct"] = df["predicted_class"] == df["ground_truth_class"]
    print(f"Correct predicted number: {df['correct'].sum()} / {len(df)}")
    # Return the mean correctness (accuracy) for each ground truth class
    return df.groupby("ground_truth_class")["correct"].mean()

print("Focal Loss:")
accuracy_macro = compute_accuracy(df_macro)
print("Weight Balanced Cross Entropy:")
accuracy_micro = compute_accuracy(df_micro)
print("Cross Entropy (Baseline):")
accuracy_weighted = compute_accuracy(df_weighted)

# Get class prevalence (frequency) from the baseline DataFrame and determine the display order
class_prevalence = df_weighted['ground_truth_class'].value_counts()
common_order = class_prevalence.index

# Function to plot a stacked bar chart comparing per-class accuracies and overlay class counts on a log scale
def plot_stacked_compare(accuracy_baseline, accuracy_improved,
                         class_counts, order,
                         label_baseline="Baseline",
                         label_improved="Improved",
                         title="Stacked Accuracy Comparison"):
    """
    Plot a stacked bar chart comparing per-class accuracies between a baseline and an improved method.
    The baseline accuracy is the bottom bar and the difference (improved - baseline) is stacked on top.
    A line plot (secondary y-axis) shows the number of samples per class on a log scale.

    Parameters:
      accuracy_baseline: pd.Series of baseline accuracies (index is class name/ID)
      accuracy_improved: pd.Series of improved accuracies
      class_counts:      pd.Series or array with sample counts for each class
      order:             Ordered list/index of classes to display
      label_baseline:    Label for the baseline bar
      label_improved:    Label for the stacked (improved) bar
      title:             Plot title
    """
    # Reindex series to the specified order
    acc_base_ordered = accuracy_baseline.reindex(order)
    acc_imp_ordered  = accuracy_improved.reindex(order)
    counts_ordered   = class_counts.reindex(order) if hasattr(class_counts, 'reindex') else np.array(class_counts)[order]

    # Calculate the difference (improved - baseline)
    diff = acc_imp_ordered - acc_base_ordered

    x = np.arange(len(order))
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Plot baseline accuracy as the bottom bar
    ax1.bar(
        x,
        acc_base_ordered.values,
        color='orange',
        edgecolor='black',
        alpha=0.7,
        label=label_baseline
    )
    # Plot the difference as a stacked bar on top of the baseline
    ax1.bar(
        x,
        diff.values,
        bottom=acc_base_ordered.values,
        color='steelblue',
        edgecolor='black',
        alpha=0.7,
        label=label_improved
    )

    # Configure left y-axis for accuracy
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(order, rotation=45, ha="right")
    ax1.legend(loc="upper right")
    ax1.set_title(title)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Create a secondary y-axis for the class sample counts
    ax2 = ax1.twinx()
    ax2.set_ylabel("#Images")
    # Plot the sample counts as a black line with markers
    ax2.plot(x, counts_ordered, color='black', linestyle='-')

    # Set the secondary y-axis to a logarithmic scale
    ax2.set_yscale('log')
    # Determine a suitable lower bound (avoid zero to prevent log(0) issues)
    if (counts_ordered > 0).any():
        ymin = counts_ordered[counts_ordered > 0].min()
    else:
        ymin = 1
    ax2.set_ylim(ymin, counts_ordered.max()*1.2)

    # Set major ticks to powers of 10 and format them as 10^n
    ax2.yaxis.set_major_locator(LogLocator(base=10))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'$10^{{{int(np.log10(y))}}}$' if y > 0 else "0"))

    plt.tight_layout()
    plt.show()

# Plot "Cross Entropy vs Focal Loss" comparison
plot_stacked_compare(
    accuracy_baseline=accuracy_weighted,
    accuracy_improved=accuracy_macro,
    class_counts=class_prevalence,
    order=common_order,
    label_baseline="Cross (Baseline)",
    label_improved="Focal Loss",
    title="Cross Entropy vs Focal Loss Under AdamW"
)

# Plot "Cross Entropy vs Weighted Cross Entropy" comparison
plot_stacked_compare(
    accuracy_baseline=accuracy_weighted,
    accuracy_improved=accuracy_micro,
    class_counts=class_prevalence,
    order=common_order,
    label_baseline="Cross (Baseline)",
    label_improved="Weighted CE",
    title="Cross Entropy vs Weighted Cross Entropy Under AdamW"
)
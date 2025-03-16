import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, FuncFormatter
from matplotlib.patches import Patch

# File paths for JSON results
base_path = r'/Users/zehualiu/Documents/GitHub/Species-Classification/test_result/weightedCrossEntropy_49.json'
improve_path = r'/Users/zehualiu/Documents/GitHub/Species-Classification/test_result/forcal_loss_49.json'

# Function to load JSON data into a Pandas DataFrame
def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

# Load data for different loss functions
df_base = load_json(base_path)          # Data for baseline
df_improve = load_json(improve_path)    # Data for improvements


# Function to compute per-class accuracy
def compute_accuracy(df):
    # Create a boolean column 'correct' indicating if the predicted class equals the ground truth
    df["correct"] = df["predicted_class"] == df["ground_truth_class"]
    print(f"Correct predicted number: {df['correct'].sum()} / {len(df)}")
    # Return the mean correctness (accuracy) for each ground truth class
    return df.groupby("ground_truth_class")["correct"].mean()

print("Cross Entropy (Baseline):")
accuracy_base = compute_accuracy(df_base)
print("Weight Balanced Cross Entropy:")
accuracy_improve = compute_accuracy(df_improve)

# Get class prevalence (frequency) from the baseline DataFrame and determine the display order
class_prevalence = df_base['ground_truth_class'].value_counts()
# Filter out classes with zero prevalence
class_prevalence = class_prevalence[class_prevalence != 0]
common_order = class_prevalence.index

# Function to plot a stacked bar chart comparing per-class accuracies with a log-scale overlay of sample counts
def plot_stacked_compare(accuracy_baseline, accuracy_improved, class_counts, order,
                         label_baseline="Baseline",
                         label_improved="Improved",
                         title="Stacked Accuracy Comparison"):
    """
    Plot a stacked bar chart comparing per-class accuracies between a baseline and an improved method.
    The baseline accuracy is displayed as the bottom bar, and the difference (improved - baseline) is stacked on top.
    A line plot (on a secondary y-axis) shows the number of samples per class on a log scale.

    Parameters:
      accuracy_baseline: pd.Series of baseline accuracies (indexed by class name/ID)
      accuracy_improved: pd.Series of improved accuracies
      class_counts:      pd.Series or array with sample counts for each class
      order:             Ordered list/index of classes to display
      label_baseline:    Label for the baseline bar
      label_improved:    Label for the difference bar (improved - baseline)
      title:             Plot title
    """
    # Reindex series to the specified order
    acc_base_ordered = accuracy_baseline.reindex(order)
    acc_imp_ordered  = accuracy_improved.reindex(order)
    counts_ordered   = class_counts.reindex(order) if hasattr(class_counts, 'reindex') else np.array(class_counts)[order]

    # Calculate the difference (improved - baseline)
    diff = acc_imp_ordered - acc_base_ordered

    # Set color based on difference: positive diff uses steelblue, negative diff uses crimson
    diff_colors = ['steelblue' if d >= 0 else 'crimson' for d in diff.values]

    x = np.arange(len(order))
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Plot baseline accuracy as the bottom bar
    ax1.bar(x, acc_base_ordered.values,
            color='orange',
            edgecolor='black',
            alpha=0.7,
            label=label_baseline)

    # Plot the difference as a stacked bar on top of the baseline
    ax1.bar(x, diff.values,
            bottom=acc_base_ordered.values,
            color=diff_colors,
            edgecolor='black',
            alpha=0.7,
            label=label_improved)

    # Configure the left y-axis for accuracy
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(order, rotation=45, ha="right")
    ax1.set_title(title)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Create custom legend items for baseline, positive diff, and negative diff
    baseline_patch = Patch(facecolor='orange', edgecolor='black', label='Adam (Baseline)')
    pos_patch = Patch(facecolor='steelblue', edgecolor='black', label='AdamW (Improved)')
    neg_patch = Patch(facecolor='crimson', edgecolor='black', label='Negative Difference')
    ax1.legend(handles=[baseline_patch, pos_patch, neg_patch], loc="upper right", framealpha=0.5)

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
    plt.savefig('per_class_comp.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.show()


# Plot comparison
plot_stacked_compare(
    accuracy_baseline=accuracy_base,
    accuracy_improved=accuracy_improve,
    class_counts=class_prevalence,
    order=common_order,
    label_baseline="Adam",
    label_improved="AdamW",
    title="Adam vs AdamW Comparison (Cross Entropy)"
)


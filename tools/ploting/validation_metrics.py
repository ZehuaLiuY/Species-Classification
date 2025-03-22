import json
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

precision_path = './val_precision.json'
recall_path = './val_recall.json'
f1_path = './val_f1.json'

with open(precision_path, "r") as f:
    precision_data = json.load(f)

with open(recall_path, "r") as f:
    recall_data = json.load(f)

with open(f1_path, "r") as f:
    f1_data = json.load(f)

precision_x_values = [item[1] for item in precision_data]
precision_y_values = [item[2] for item in precision_data]
recall_x_values = [item[1] for item in recall_data]
recall_y_values = [item[2] for item in recall_data]
f1_x_values = [item[1] for item in f1_data]
f1_y_values = [item[2] for item in f1_data]

# using Savitzky-Golay filter to smooth the data
window_length = 11
polyorder = 3
smoothed_precision = savgol_filter(precision_y_values, window_length, polyorder)
smoothed_recall = savgol_filter(recall_y_values, window_length, polyorder)
smoothed_f1 = savgol_filter(f1_y_values, window_length, polyorder)

fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True)

# unsmoothed
axes[0, 0].plot(precision_x_values, precision_y_values, linestyle='-', color='blue')
axes[0, 0].set_ylabel("Precision")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_title("Precision")
axes[0, 0].grid(True)

axes[0, 1].plot(recall_x_values, recall_y_values, linestyle='-', color='green')
axes[0, 1].set_ylabel("Recall")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_title("Recall")
axes[0, 1].grid(True)

axes[0, 2].plot(f1_x_values, f1_y_values, linestyle='-', color='red')
axes[0, 2].set_ylabel("F1")
axes[0, 2].set_xlabel("Epoch")
axes[0, 2].set_title("F1")
axes[0, 2].grid(True)

# smoothed
axes[1, 0].plot(precision_x_values, smoothed_precision, linestyle='-', color='blue')
axes[1, 0].set_ylabel("Smoothed Precision")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_title("Smoothed Precision")
axes[1, 0].grid(True)

axes[1, 1].plot(recall_x_values, smoothed_recall, linestyle='-', color='green')
axes[1, 1].set_ylabel("Smoothed Recall")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_title("Smoothed Recall")
axes[1, 1].grid(True)

axes[1, 2].plot(f1_x_values, smoothed_f1, linestyle='-', color='red')
axes[1, 2].set_ylabel("Smoothed F1")
axes[1, 2].set_xlabel("Epoch")
axes[1, 2].set_title("Smoothed F1")
axes[1, 2].grid(True)

for ax in axes.flatten():
    ax.axvline(x=26, color='grey', linestyle='--', label='Early stopping Point')
    ax.legend()

plt.tight_layout()
plt.savefig('validation_metrics_smoothed.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()

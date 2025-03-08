import json
import matplotlib.pyplot as plt

prcision_path = './val_precision.json'
recall_path = './val_recall.json'
f1_path = './val_f1.json'

with open(prcision_path, "r") as f:
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

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

# precision plot
ax1.plot(precision_x_values, precision_y_values, linestyle='-', color='blue')
ax1.set_ylabel("Precision")
ax1.set_xlabel("Epoch")
ax1.set_title("Precision")
ax1.grid(True)

# recall plot
ax2.plot(recall_x_values, recall_y_values, linestyle='-', color='green')
ax2.set_ylabel("Recall")
ax2.set_xlabel("Epoch")
ax2.set_title("Recall")
ax2.grid(True)

# f1 plot
ax3.plot(f1_x_values, f1_y_values, linestyle='-', color='red', label='F1')
ax3.set_ylabel("F1")
ax3.set_xlabel("Epoch")
ax3.set_title("F1")
ax3.grid(True)

plt.savefig('validation_metrics.pdf', dpi=600, bbox_inches='tight', pad_inches=0)

plt.show()
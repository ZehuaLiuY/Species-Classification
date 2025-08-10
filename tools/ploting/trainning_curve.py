import json
import matplotlib.pyplot as plt
import numpy as np

train_path = './train_acc.json'
val_path = './val_acc.json'

with open(train_path, "r") as f:
    train_data = json.load(f)

with open(val_path, "r") as f:
    val_data = json.load(f)

train_x_values = [item[1] for item in train_data]
train_y_values = [item[2] for item in train_data]

val_x_values = [item[1] for item in val_data]
val_y_values = [item[2] for item in val_data]

diff = np.array(train_y_values) - np.array(val_y_values)
# diff = np.array(val_y_values) - np.array(train_y_values)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

# train/val accuracy plot
# train acc colour # 1d73b6
# val acc colour # 24a645
# train loss colour # 8768a6
# val loss colour # 6dc2dc
ax1.plot(train_x_values, train_y_values, linestyle='-', color='#1d73b6', label='Train Accuracy')
ax1.plot(val_x_values, val_y_values, linestyle='-', color='#24a645', label='Validation Accuracy')
ax1.axvline(x=21, color='grey', linestyle='--', label='Early Stopping Point')
ax1.axhline(0.90, linestyle=':', linewidth=1, color='grey', label='90% Threshold')

# for spine in ['top', 'right']:
#     ax1.spines[spine].set_visible(False)
ax1.spines['bottom'].set_linewidth(0.5)
ax1.spines['left'].set_linewidth(0.8)

ax1.set_ylabel("Loss")
ax1.set_xlabel("Epochs")
ax1.set_title("Train/Validation Accuracy")
ax1.legend(loc='upper left')
ax1.grid(True)

# difference plot
ax2.plot(train_x_values, diff, linestyle='-', color='red', label='Difference (Train - Validation)')
ax2.axvline(x=21, color='grey', linestyle='--', label='Early Stopping Point')
# ax2.spines['bottom'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
ax2.axhline(y=0, color='black', linestyle='-')
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy Difference")
ax2.set_title("Difference between Train and Validation Accuracy")
ax2.legend(loc='upper left')
ax2.grid(True)

plt.savefig('scheduler_acc_curve.pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)

plt.show()


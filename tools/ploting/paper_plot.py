import json
import matplotlib.pyplot as plt
import numpy as np

train_path = './train_acc.json'
val_path   = './val_acc.json'
with open(train_path, "r") as f:
    train_data = json.load(f)
with open(val_path, "r") as f:
    val_data   = json.load(f)

train_x = [item[1] for item in train_data]
train_y = [item[2] for item in train_data]
val_x   = [item[1] for item in val_data]
val_y   = [item[2] for item in val_data]

diff = np.array(val_y) - np.array(train_y)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)


ax1.plot(train_x, train_y,
         linestyle='-', marker='o', markersize=5, linewidth=1.2,
         color='#1d73b6', label='Train Accuracy')

ax1.plot(val_x, val_y,
         linestyle='--', marker='s', markersize=5, linewidth=1.2,
         color='#24a645', label='Validation Accuracy')

ax1.axvline(21, linestyle='--', linewidth=1, color='grey', label='Early Stopping')
ax1.axhline(0.90, linestyle=':', linewidth=1, color='grey', label='90% Threshold')

for spine in ['top','right']:
    ax1.spines[spine].set_visible(False)
ax1.spines['bottom'].set_linewidth(0.5)
ax1.spines['left'].set_linewidth(0.8)

ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
ax1.grid(axis='y', linestyle=':', linewidth=0.5)

ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.set_title("Train vs. Validation Accuracy")
ax1.legend(loc='lower right')


ax2.plot(train_x, diff,
         linestyle='--', marker='o', markersize=5, linewidth=1.2,
         color='red', label='Val − Train')
ax2.axvline(21, linestyle='--', linewidth=1, color='grey', label='Early Stopping')
ax2.axhline(0, linestyle='-', linewidth=1.2, color='black', label='Zero Difference')

for spine in ['top','right']:
    ax2.spines[spine].set_visible(False)
ax2.spines['bottom'].set_linewidth(0.5)
ax2.spines['left'].set_linewidth(0.8)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')
ax2.grid(axis='y', linestyle=':', linewidth=0.5)

ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy Difference")
ax2.set_title("Difference between Val and Train")
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()
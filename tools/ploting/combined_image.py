import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

image_folder = 'image'
image_files = [
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if os.path.isfile(os.path.join(image_folder, f))
       and f.lower().endswith(('.jpg', '.jpeg', '.png'))
]
image_files.sort()

cols = 6
rows = (len(image_files) + cols - 1) // cols

fig, axes = plt.subplots(
    rows, cols,
    figsize=(20, rows * 2),
    gridspec_kw={'wspace':0, 'hspace':0}
)
axes = axes.flatten()

for ax, img_file in zip(axes, image_files):
    img = mpimg.imread(img_file)
    ax.imshow(img, aspect='auto')
    ax.axis('off')
    ax.margins(0.1)

for ax in axes[len(image_files):]:
    ax.remove()

plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=1, hspace=1)

plt.savefig('combined_image.pdf', dpi=300, bbox_inches='tight', pad_inches=0)

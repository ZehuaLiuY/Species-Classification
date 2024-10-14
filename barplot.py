import json
import pandas as pd
import matplotlib.pyplot as plt

with open('part3output.json') as f:
    data = json.load(f)

detections = []
for entry in data['images']:
    for detection in entry['detections']:
        detections.append({
            'category': detection['category'],
            'confidence': detection['conf']
        })

df = pd.DataFrame(detections)

category_counts = df['category'].value_counts()

category_counts.plot(kind= 'bar')
plt.title('Category Counts')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

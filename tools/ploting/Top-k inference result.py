import json, pandas as pd
from collections import Counter

json_path = r'/Users/zehualiu/Desktop/test_result/unbiased_test_results/json/48/LADM_sc.json'
data = json.loads(open(json_path, encoding='utf-8').read())

filtered = [e for e in data if e['ground_truth_class'] == 'wild turkey']
cnt      = Counter(e['predicted_class'] for e in filtered)

df = (pd.DataFrame(cnt.items(), columns=['Predicted class', 'Count'])
      .sort_values('Count', ascending=False))
df['Percent'] = df['Count'] / df['Count'].sum() * 100

TOP_K = 15
if len(df) > TOP_K:
    other_cnt = df.iloc[TOP_K:]['Count'].sum()
    other_row = pd.DataFrame({
        'Predicted class': ['Other'],
        'Count':          [other_cnt],
        'Percent':        [other_cnt / cnt.total() * 100]
    })

    df = pd.concat([df.iloc[:TOP_K], other_row], ignore_index=True)

print(df.to_latex(index=False, float_format='%.1f'))

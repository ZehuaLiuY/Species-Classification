import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

macro_path = r'/Users/zehualiu/Documents/GitHub/Project-Prep/test_result/46/results_macro.json'
micro_path = r'/Users/zehualiu/Documents/GitHub/Project-Prep/test_result/46/results_micro.json'
weighted_path = r'/Users/zehualiu/Documents/GitHub/Project-Prep/test_result/46/results_weighted.json'

focal_loss_path = r'/Users/zehualiu/Documents/GitHub/Project-Prep/test_result/forcal_loss_49.json'
weighted_cross_path = r'/Users/zehualiu/Documents/GitHub/Project-Prep/test_result/weightedCrossEntropy_49.json'
cross_path = r'/Users/zehualiu/Documents/GitHub/Project-Prep/test_result/results_weighted.json'

def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

df_macro = load_json(focal_loss_path)     # focal
df_micro = load_json(weighted_cross_path) # weighted cross
df_weighted = load_json(cross_path)       # cross (baseline)

def compute_accuracy(df):
    df["correct"] = df["predicted_class"] == df["ground_truth_class"]
    print(f"correct predicted number: {df['correct'].sum()} / {len(df)}")
    # 返回每个类的准确率
    return df.groupby("ground_truth_class")["correct"].mean()

print("Focal Loss:")
accuracy_macro = compute_accuracy(df_macro)
print("Weight Balanced Cross Entropy:")
accuracy_micro = compute_accuracy(df_micro)
print("Cross Entropy (Baseline):")
accuracy_weighted = compute_accuracy(df_weighted)

# 根据baseline df来获取类出现频次并排序
class_prevalence = df_weighted['ground_truth_class'].value_counts()
common_order = class_prevalence.index


def plot_stacked_compare(accuracy_baseline, accuracy_improved,
                         class_counts,
                         order,
                         label_baseline="Baseline",
                         label_improved="Improved",
                         title="Stacked Accuracy Comparison"):
    """
    在同一张图上用堆叠柱状图对比 baseline 和 improved 的 per-class accuracy，
    其中 baseline 在下面，(improved - baseline) 叠在上面。
    同时在右侧 y 轴叠加 class_counts 的曲线。

    参数：
      accuracy_baseline: pd.Series, index是类名/ID
      accuracy_improved: pd.Series, 同上
      class_counts:      pd.Series or array，每个类出现频次
      order:             要显示的类顺序 (list或index)
      label_baseline:    底部柱子的图例名称
      label_improved:    叠加部分柱子的图例名称
      title:             图标题
    """

    # 根据指定顺序 reindex
    acc_base_ordered = accuracy_baseline.reindex(order)
    acc_imp_ordered  = accuracy_improved.reindex(order)
    counts_ordered   = class_counts.reindex(order) if hasattr(class_counts, 'reindex') else np.array(class_counts)[order]

    # 计算“叠加”部分：diff = improved - baseline
    diff = acc_imp_ordered - acc_base_ordered

    x = np.arange(len(order))
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # 第1段：baseline 柱子
    ax1.bar(
        x,
        acc_base_ordered.values,
        color='orange',
        edgecolor='black',
        alpha=0.7,
        label=label_baseline
    )
    # 第2段：叠加在 baseline 上的 “diff”
    ax1.bar(
        x,
        diff.values,
        bottom=acc_base_ordered.values,
        color='steelblue',
        edgecolor='black',
        alpha=0.7,
        label=label_improved
    )

    # 左侧 y 轴：accuracy
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(order, rotation=45, ha="right")
    ax1.legend(loc="upper right")
    ax1.set_title(title)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # 右侧 y 轴：类的样本数
    ax2 = ax1.twinx()
    ax2.set_ylabel("#Images")
    ax2.plot(x, counts_ordered, color='black', linestyle='-')
    ax2.set_ylim(0, counts_ordered.max()*1.2)

    plt.tight_layout()
    plt.show()


# accuracy_weighted 是 Cross (Baseline)
# accuracy_macro    是 Focal Loss
# accuracy_micro    是 Weighted CE
# class_prevalence  是每个类的样本数
# common_order      是类的排序

# 画一张 “Cross vs Focal” 的堆叠图
plot_stacked_compare(
    accuracy_baseline=accuracy_weighted,
    accuracy_improved=accuracy_macro,
    class_counts=class_prevalence,
    order=common_order,
    label_baseline="Cross (Baseline)",
    label_improved="Focal Loss",
    title="Cross Entropy vs Focal Loss Under AdamW"
)

# 画一张 “Cross vs Weighted” 的堆叠图
plot_stacked_compare(
    accuracy_baseline=accuracy_weighted,
    accuracy_improved=accuracy_micro,
    class_counts=class_prevalence,
    order=common_order,
    label_baseline="Cross (Baseline)",
    label_improved="Weighted CE",
    title="Cross Entropy vs Weighted Cross Entropy Under AdamW"
)
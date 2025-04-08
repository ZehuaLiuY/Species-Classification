import re

def parse_metrics_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    metrics = []
    pattern = re.compile(
        r'^Class\s+(\d+)\s+\((.*?)\)\s+([\d\.]+)\s+(\d+)\s+(\d+)\s+([\d\.]+)\s+(\d+)\s+([\d\.]+)'
    )

    table_started = False
    for line in lines:
        line = line.strip()
        if not table_started:
            if line.startswith('-----'):
                table_started = True
            continue

        match = pattern.match(line)
        if match:
            class_id   = int(match.group(1))
            class_name = match.group(2)
            precision  = float(match.group(3))
            true_pos   = int(match.group(4))
            class_bias = int(match.group(5))
            recall     = float(match.group(6))
            prevalence = int(match.group(7))
            f1         = float(match.group(8))
            metrics.append({
                'class_id': class_id,
                'class_name': class_name,
                'precision': precision,
                'true_positive': true_pos,
                'class_bias': class_bias,
                'recall': recall,
                'prevalence': prevalence,
                'f1': f1
            })
    return metrics

def compute_weighted_accuracy(metrics, selected_classes):
    sum_true_pos = 0
    sum_prevalence = 0
    for m in metrics:
        if m['class_id'] in selected_classes:
            sum_true_pos += m['true_positive']
            sum_prevalence += m['prevalence']
    if sum_prevalence == 0:
        return 0
    return sum_true_pos / sum_prevalence

if __name__ == '__main__':
    file_path = r"G:\Code\github\Project-Prep\test_result\txt\48\LDAM_sc.txt "

    selected_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 17, 18, 19, 20, 22, 23, 25,
                        26, 27, 28, 30, 31, 33, 34, 35, 36, 37, 39, 41, 43, 44, 46, 47,
                        ]

    metrics = parse_metrics_file(file_path)
    if not metrics:
        print("cannot find any metrics in the file.")
    else:
        weighted_accuracy = compute_weighted_accuracy(metrics, selected_classes)
        print(f"The weighted accuracy is: {weighted_accuracy:.4f}")

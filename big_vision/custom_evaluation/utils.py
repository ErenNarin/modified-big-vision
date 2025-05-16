def normalize(metric_name, value):
    norm_value = 0
    if metric_name == 'wmd':  # often in [0, 0.50], map to [0, 1]
        norm_value = max((1 - value), 0.0)
    elif metric_name == 'bertscore':  # often in [0.75, 0.95], map to [0, 1]
        norm_value = min(1, (max(0, (float(value) - 0.75)) * 5))
    elif metric_name == 'rogue':  # already in [0, 1], get average score
        sum = 0
        for m, s in value.items():
            sum += s.fmeasure
        norm_value = sum / len(value)
    elif metric_name == 'grammar':  # >= 2 errors will get minimum score
        norm_value = max(0, 1 - value / 2)
    elif metric_name == 'readability':  # already in [0, 1], map to [0, 1]
        norm_value = min(max(value / 100, 0), 1)
    return norm_value


def compute_hybrid_score(metrics, weights):
    total = 0.0
    for k, v in metrics.items():
        total += weights[k] * normalize(k, v)
    return total

# Convert to COCO-style format for pycocoevalcap
def format_for_coco(data):
    return {k: [v] if not isinstance(v, list) else v for k, v in data.items()}


def run_coco_metric(metric_class, gts, res):
    scorer = metric_class()
    score, _ = scorer.compute_score(gts, res)
    return score


def normalize(metric_name, value):
    norm_value = 0
    if metric_name == 'cider':  # usually in [0, 10], normalize to [0, 1]
        norm_value = min(float(value) / 10, 1.0)
    elif metric_name == 'spice':  # already in [0, 1]
        norm_value = value
    elif metric_name == 'bertscore':  # mostly in [0.75, 0.95], map to [0, 1]
        return min(1, (max(0.0, (float(value) - 0.75)) * 5))
    elif metric_name == 'bleu':  # often in [0, 1], get average score
        sum = 0
        for v in value:
            sum += v
        norm_value = sum / len(value)
    elif metric_name == 'rogue':  # already in [0, 1], get average score
        sum = 0
        for m, s in value.items():
            sum += s.fmeasure
        norm_value = sum / len(value)
    elif metric_name == 'grammar':  # fewer errors = better
        norm_value = max(0, 1 - value / 2)  # >= 2 errors will get minimum score
    elif metric_name == 'readability':  # Flesch: (0–100) -> [0-1]
        norm_value = min(max(value / 100, 0), 1)
    return norm_value


def compute_hybrid_score(metrics, weights):
    total = 0.0
    for k, v in metrics.items():
        total += weights[k] * normalize(k, v)
    return total

import math
from sklearn.metrics import f1_score
from typing import List, Dict

def compute_instance_metrics(
    distributions: List[Dict[str, float]],
    true_labels: List[str],
    label_options: List[str],
    f1_average: str = "macro"
) -> tuple:
    """
    Computes instance-level Cross-Entropy (NLL), Brier Score, and F1 score, and then averages them over the dataset.

    Args:
        distributions: A list of dicts, each mapping label_option -> probability.
        true_labels:   A list of ground-truth labels.
        label_options: A list of all possible label strings.
        f1_average:    The averaging method for F1 ("macro", "micro", "weighted", etc.).

    Returns:
        A tuple containing:
            - avg_cross_entropy: float
            - avg_brier_score: float
            - f1_value: float
    """
    cross_entropies = []
    brier_scores = []
    predicted_labels = []

    for dist_dict, true_label in zip(distributions, true_labels):
        # 1. cross-entropy / negative log-likelihood
        prob_true = dist_dict.get(true_label, 0.0)
        nll = -math.log(prob_true + 1e-12)  # avoid log(0)
        cross_entropies.append(nll)

        # 2. brier score
        brier_sum = 0.0
        for lbl, p in dist_dict.items():
            y = 1.0 if lbl == true_label else 0.0
            brier_sum += (p - y) ** 2
        # turn missing labels to probabilities as 0
        missing_labels = set(label_options) - set(dist_dict.keys())
        for lbl in missing_labels:
            y = 1.0 if lbl == true_label else 0.0
            brier_sum += (0.0 - y) ** 2
        brier_scores.append(brier_sum)

        # 3. argmax to get pred label
        best_label = max(dist_dict, key=dist_dict.get)
        predicted_labels.append(best_label)

    avg_cross_entropy = float(np.mean(cross_entropies))
    avg_brier_score = float(np.mean(brier_scores))
    f1_val = f1_score(true_labels, predicted_labels, average=f1_average)

    return avg_cross_entropy, avg_brier_score, f1_val


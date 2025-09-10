import re
import string
import time
import math
from collections import Counter
from typing import Callable, Iterable, List, Tuple, Dict, Any


# -----------------------
# Text normalization utils
# -----------------------
def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation/articles/extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return text.translate(str.maketrans('', '', string.punctuation))
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# ---------------
# Core QA metrics
# ---------------
def compute_em(preds: List[str], golds: List[str]) -> float:
    correct = sum(
        normalize_answer(p) == normalize_answer(g)
        for p, g in zip(preds, golds)
    )
    return correct / len(golds) if golds else 0.0


def compute_f1(preds: List[str], golds: List[str]) -> float:
    def f1_score(pred: str, gold: str) -> float:
        pred_tokens = normalize_answer(pred).split()
        gold_tokens = normalize_answer(gold).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            return float(pred_tokens == gold_tokens)
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        return (2 * precision * recall) / (precision + recall)

    scores = [f1_score(p, g) for p, g in zip(preds, golds)]
    return sum(scores) / len(scores) if scores else 0.0


# Alias: in SimpleQA, accuracy == EM after normalization
def compute_accuracy(preds: List[str], golds: List[str]) -> float:
    return compute_em(preds, golds)


# ----------------
# Latency metrics
# ----------------
def latency_stats(latencies: List[float]) -> Dict[str, float]:
    """Return avg and p95 latency using nearest-rank percentile (no NumPy)."""
    if not latencies:
        return {"avg_latency": 0.0, "p95_latency": 0.0}
    avg = sum(latencies) / len(latencies)
    xs = sorted(latencies)
    rank = max(1, math.ceil(0.95 * len(xs)))  # nearest-rank p95
    p95 = xs[rank - 1]
    return {"avg_latency": float(avg), "p95_latency": float(p95)}


# -------------------------
# End-to-end eval utilities
# -------------------------
def time_predictions(
    answer_fn: Callable[[str], str],
    data: Iterable[Dict[str, Any]],
) -> Tuple[List[str], List[str], List[float]]:
    """
    Run inference over `data`, timing each call to `answer_fn(question)`.
    Returns (predictions, golds, latencies).
    """
    predictions, golds, latencies = [], [], []
    for ex in data:
        q = ex["question"]
        g = ex["answer"]

        start = time.perf_counter()
        pred = answer_fn(q)
        end = time.perf_counter()

        predictions.append(pred)
        golds.append(g)
        latencies.append(end - start)

    return predictions, golds, latencies


def evaluate(
    preds: List[str],
    golds: List[str],
    latencies: List[float],
) -> Dict[str, float]:
    """Compute all summary metrics in one place."""
    em = compute_em(preds, golds)
    f1 = compute_f1(preds, golds)
    acc = compute_accuracy(preds, golds)
    lat = latency_stats(latencies)
    return {"accuracy": acc, "em": em, "f1": f1, **lat}


def run_eval(
    answer_fn: Callable[[str], str],
    data: Iterable[Dict[str, Any]],
) -> Tuple[List[str], List[str], List[float], Dict[str, float]]:
    """
    Convenience wrapper:
    - times predictions
    - computes all metrics
    - returns (preds, golds, latencies, metrics_dict)
    """
    preds, golds, lats = time_predictions(answer_fn, data)
    metrics = evaluate(preds, golds, lats)
    return preds, golds, lats, metrics

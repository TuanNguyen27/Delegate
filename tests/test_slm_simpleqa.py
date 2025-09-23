# test_slm_simpleqa.py
from omegaconf import OmegaConf

from src.data.simpleqa_loader import load_simpleqa
from src.models.slm import slm_answer
from src.eval.metrics import run_eval


def main():
    # 1. Load config
    cfg = OmegaConf.load("configs/simpleqa_slm.yaml")

    # 2. Load dataset
    data = list(load_simpleqa(cfg))

    # 3. Wrap engine into a function that matches run_eval’s signature
    engine = lambda q: slm_answer(q, cfg)

    # 4. Run evaluation
    preds, golds, lats, metrics = run_eval(engine, data)

    # 5. Print sample outputs (first 5)
    for q, g, p in zip([ex["question"] for ex in data[:5]],
                       [ex["answer"] for ex in data[:5]],
                       preds[:5]):
        print("="*40)
        print(f"Q: {q}")
        print(f"Gold: {g}")
        print(f"Pred: {p}")

    # 6. Print aggregate metrics
    print("\n=== Aggregate Metrics ===")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"EM:       {metrics['em']:.3f}")
    print(f"F1:       {metrics['f1']:.3f}")
    print(f"Latency:  avg={metrics['avg_latency']:.3f}s, p95={metrics['p95_latency']:.3f}s")


if __name__ == "__main__":
    main()

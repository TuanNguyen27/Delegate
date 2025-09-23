# test_llm_simpleqa.py
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.data.simpleqa_loader import load_simpleqa
from src.models.llm import llm_answer
from src.eval.metrics import run_eval


def main():
    # 1. Load environment (API key)
    load_dotenv()

    # 2. Load config
    cfg = OmegaConf.load("configs/simpleqa_llm.yaml")

    # 3. Load dataset
    data = list(load_simpleqa(cfg))

    # 4. Wrap engine into a function that matches run_eval’s signature
    engine = lambda q: llm_answer(q, cfg)

    # 5. Run evaluation
    preds, golds, lats, metrics = run_eval(engine, data)

    # 6. Print sample outputs (first 5)
    for q, g, p in zip([ex["question"] for ex in data[:5]],
                       [ex["answer"] for ex in data[:5]],
                       preds[:5]):
        print("="*40)
        print(f"Q: {q}")
        print(f"Gold: {g}")
        print(f"Pred: {p}")

    # 7. Print aggregate metrics
    print("\n=== Aggregate Metrics ===")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"EM:       {metrics['em']:.3f}")
    print(f"F1:       {metrics['f1']:.3f}")
    print(f"Latency:  avg={metrics['avg_latency']:.3f}s, p95={metrics['p95_latency']:.3f}s")


if __name__ == "__main__":
    main()

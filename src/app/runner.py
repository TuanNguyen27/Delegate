# src/app/runner.py
import argparse
from omegaconf import OmegaConf

from dataio.gsm8k_loader import load_gsm8k
from dataio.simpleqa_loader import load_simpleqa
from engines.slm import slm_answer
from eval.metrics import compute_accuracy  # placeholder, replace with EM/F1 later

def main():
    # 1. Parse CLI args
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    # 2. Load config
    cfg = OmegaConf.load(args.config)

    # 3. Pick dataset
    if cfg.dataset == "openai/gsm8k":
        data = load_gsm8k(cfg)
    elif cfg.dataset == "simpleqa":
        data = load_simpleqa(cfg)
    else:
        raise ValueError(f"Unknown dataset {cfg.dataset}")

    preds, golds = [], []
    # 4. Loop through examples
    for ex in data:
        pred = slm_answer(ex["question"])   # engine stub
        preds.append(pred)
        golds.append(ex["answer"])

    # 5. Evaluate
    acc = compute_accuracy(preds, golds)
    print(f"Accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()

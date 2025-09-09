# src/app/run_router.py
from omegaconf import OmegaConf
from datasets.gsm8k import load_gsm8k
from datasets.simpleqa import load_simpleqa
from engines.slm import slm_answer
from eval.metrics import compute_accuracy

def main():
    # 1. Load config
    cfg = OmegaConf.load("configs/default.yaml")

    # 2. Pick dataset
    if cfg.dataset == "openai/gsm8k":
        data = load_gsm8k(cfg)
    elif cfg.dataset == "simpleqa":
        data = load_simpleqa(cfg)å
    else:
        raise ValueError(f"Unknown dataset {cfg.dataset}")

    preds, golds = [], []
    # 3. Loop through examples
    for ex in data:
        pred = slm_answer(ex["question"])   # engine
        preds.append(pred)
        golds.append(ex["answer"])

    # 4. Evaluate
    acc = compute_accuracy(preds, golds)
    print(f"Accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()

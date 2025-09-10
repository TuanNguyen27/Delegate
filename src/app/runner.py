# src/app/runner.py
import argparse
from omegaconf import OmegaConf

from src.data.gsm8k_loader import load_gsm8k
from src.data.simpleqa_loader import load_simpleqa
from models.slm import slm_answer # For SLM (to be put in later)
from eval.metrics import compute_accuracy  # placeholder, replace with EM/F1 later

def main():
    # Parse CLI args
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    # Load config from argparse 
    cfg = OmegaConf.load(args.config)

    # Pick dataset
    if cfg.dataset == "openai/gsm8k":
        data = load_gsm8k(cfg)
    elif cfg.dataset == "simpleqa":
        data = load_simpleqa(cfg)
    else:
        raise ValueError(f"Unknown dataset {cfg.dataset}")

    predictions, golds = [], [] # Lists of predictions and ground truths ("golds")
    # Loop through examples in the dataset
    for ex in data:
        prediction = slm_answer(ex["question"])   # engine stub
        predictions.append(prediction)
        golds.append(ex["answer"]) 

    # Evaluate
    acc = compute_accuracy(predictions, golds)
    print(f"Accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()
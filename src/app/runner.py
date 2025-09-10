# src/app/runner.py
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from src.data.gsm8k_loader import load_gsm8k
from src.data.simpleqa_loader import load_simpleqa
from src.models.slm import slm_answer
from src.models.llm import llm_answer
from src.eval.metrics import run_eval


def save_run_log(cfg, predictions, golds, latencies, metrics, outdir="logs/"):
    """Save per-item predictions and run manifest to disk."""
    Path(outdir).mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Per-item log (JSONL)
    run_file = Path(outdir) / f"run_{timestamp}.jsonl"
    with open(run_file, "w") as f:
        for i, (p, g, lat) in enumerate(zip(predictions, golds, latencies)):
            f.write(json.dumps({"id": i, "pred": p, "gold": g, "latency": lat}) + "\n")

    # Manifest (config + metrics)
    manifest_file = Path(outdir) / f"manifest_{timestamp}.json"
    manifest = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": metrics,
        "n": len(predictions),
        "timestamp": timestamp,
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] Logs saved to {run_file} and {manifest_file}")


def main():
    # 1) Parse CLI args
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    # 2) Load config
    cfg = OmegaConf.load(args.config)

    # 3) Pick dataset
    if cfg.dataset == "openai/gsm8k":
        data = load_gsm8k(cfg)
    elif cfg.dataset == "simpleqa":
        data = load_simpleqa(cfg)
    else:
        raise ValueError(f"Unknown dataset {cfg.dataset}")

    # 4) Pick engine (SLM vs LLM)
    if "slm" in cfg:
        engine = lambda q: slm_answer(q, cfg)
    elif "llm" in cfg:
        engine = lambda q: llm_answer(q, cfg)
    else:
        raise ValueError("Config must define either 'slm' or 'llm' block")

    # 5) Run inference + metrics
    predictions, golds, latencies, metrics = run_eval(engine, data)

    # 6) Report summary
    print(
        "Acc={accuracy:.3f}, EM={em:.3f}, F1={f1:.3f}, "
        "Latency={avg_latency:.3f}s (p95={p95_latency:.3f}s)".format(**metrics)
    )

    # 7) Save logs
    save_run_log(cfg, predictions, golds, latencies, metrics)


if __name__ == "__main__":
    # Tip: run from repo root as a module to keep imports clean:
    #   python -m src.app.runner --config configs/slm.yaml
    #   python -m src.app.runner --config configs/llm.yaml
    main()

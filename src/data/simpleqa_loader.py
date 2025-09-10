# src/data/simpleqa_loader.py

import pandas as pd
import random

def load_simpleqa(cfg):
    seed = getattr(cfg.seed, "seed", 123) # Reproducibility
    n_samples = cfg.get("n_samples", None)
    
    path = "src/data/simpleqa.csv" # Local copy
    df = pd.read_csv(path)

    num_samples = min(n_samples, len(df))
    if cfg.n_samples:
        df = df.sample(n=num_samples, random_state=seed)

    for i, row in df.iterrows():
        yield {
            "id": i,
            "question": row["problem"],
            "answer": row["answer"],
        }
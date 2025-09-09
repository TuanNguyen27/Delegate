import pandas as pd
import random

def load_simpleqa(cfg):
    # Reproducibility
    seed = getattr(cfg, "seed", 123)
    random.seed(seed)

    url = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
    df = pd.read_csv(url)

    n_samples = getattr(cfg, "n_samples", None)
    if n_samples:
        df = df.sample(n=n_samples, random_state=seed)

    for i, row in df.iterrows():
        yield {
            "id": i,
            "question": row["problem"],
            "answer": row["answer"],
        }

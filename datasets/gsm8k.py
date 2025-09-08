from datasets import load_dataset
from omegaconf import OmegaConf
import os
import random

# step 1: load config 
cfg = OmegaConf.load("configs/default.yaml")
 
# step 2 :set seed (for reproducibility)
random.seed(cfg.seed)

# step 3: load dataset
dataset = load_dataset(cfg.dataset,
                       cfg.subset,
                       split=cfg.split) 

# step 4: output dir exists
os.makedirs(cfg.output_dir, exist_ok=True)

# step 5: a small sample to a file
output_file = os.path.join(cfg.output_dir, "sample.txt")
with open(output_file, "w") as f:
    for i, example in enumerate(dataset):
        f.write(f"Example {i}:\n")
        f.write(f"Question: {example['question']}\n")
        f.write(f"Answer: {example['answer']}\n\n")

print(f"Saved {len(dataset)} examples to {output_file}")
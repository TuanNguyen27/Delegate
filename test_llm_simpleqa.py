# test_llm_simpleqa.py
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.data.simpleqa_loader import load_simpleqa
from src.models.llm import llm_answer

def main():
    # 1. Load environment (API key)
    load_dotenv()

    # 2. Load config
    cfg = OmegaConf.load("configs/simpleqa_llm.yaml")

    # 3. Load a few dataset samples
    data = list(load_simpleqa(cfg))

    # 4. Run inference on first few examples
    for ex in data[:5]:  # limit to 5 for sanity check
        q, gold = ex["question"], ex["answer"]
        pred = llm_answer(q, cfg)
        print("="*40)
        print(f"Q: {q}")
        print(f"Gold: {gold}")
        print(f"Pred: {pred}")

if __name__ == "__main__":
    main()

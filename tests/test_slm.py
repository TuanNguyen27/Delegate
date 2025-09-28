# test_slm.py
from omegaconf import OmegaConf
from src.models.slm import slm_answer

def main():
    # Load config
    cfg = OmegaConf.load("configs/slm.yaml")
    
    # Test question
    question = "Roger has 5 tennis balls. He buys 2 cans. Each can has 3 balls. How many balls are there in total?"
    answer = slm_answer(question, cfg)
    
    print("Q: " + question)
    print("A: " + answer)
    
if __name__ == "__main__":
    main()
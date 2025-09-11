# test_slm.py
from omegaconf import OmegaConf
from src.models.slm import slm_answer

def main():
    # Load config
    cfg = OmegaConf.load("configs/slm.yaml")
    
    # Test question
    question = "What is the capital of France?"
    answer = slm_answer(question, cfg)
    
    print("Q: " + question)
    print("A: " + answer)
    
if __name__ == "__main__":
    main()
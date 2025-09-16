# test_slm.py
from omegaconf import OmegaConf
from src.models.slm import slm_answer

def main():
    # Load config
    cfg = OmegaConf.load("configs/slm.yaml")
    
    # Test question
    question = "Artificial intelligence (AI) has undergone rapid development over the past decade, moving from narrow rule-based systems to large-scale neural networks capable of language understanding, image generation, and even strategic gameplay. One of the key drivers of this transformation has been the increasing availability of massive datasets, combined with advances in computational power through specialized hardware like GPUs and TPUs. As models such as GPT, BERT, and diffusion architectures became mainstream, researchers also began to uncover new challenges: issues of bias, hallucination, and interpretability. While AI has already found applications in healthcare, finance, education, and entertainment, concerns about fairness, job displacement, and safety have led to debates about how quickly society should adopt these technologies. Policymakers are now caught between encouraging innovation and enforcing safeguards, while industry leaders race to release more capable and efficient models. As we look ahead, the central question is not only what AI can do, but how humans can responsibly integrate it into their lives and institutions. Please summarize in 50 words or less. Output the number of words"
    answer = slm_answer(question, cfg)
    
    print("Q: " + question)
    print("A: " + answer)
    
if __name__ == "__main__":
    main()
# test_llm.py
from omegaconf import OmegaConf
from src.models.llm import llm_answer
from dotenv import load_dotenv
load_dotenv() 

cfg = OmegaConf.load("configs/llm.yaml")
question = "What U.S. embassy did Alex Michel work for after graduating college??"
answer = llm_answer(question, cfg)
print("Q:", question)
print("A:", answer)

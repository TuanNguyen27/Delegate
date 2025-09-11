# test_llm.py
from omegaconf import OmegaConf
from src.models.llm import llm_answer
from dotenv import load_dotenv
load_dotenv() 

cfg = OmegaConf.load("configs/llm.yaml")
question = "When was World War 1?"
answer = llm_answer(question, cfg)
print("Q:", question)
print("A:", answer)

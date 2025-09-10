# main.py
# Kinda similar to runner, but main.py is a quick sanity check
from omegaconf import OmegaConf
from src.data import load_data

def main(config_path: str):
    cfg = OmegaConf.load(config_path)
    for ex in load_data(cfg):
        print(ex)

if __name__ == "__main__":
    main("configs/gsm8k.yaml")
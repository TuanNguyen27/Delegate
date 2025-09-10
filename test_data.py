# test_data.py
from omegaconf import OmegaConf
from src.data import load_data

def data_test(config_path: str):
    cfg = OmegaConf.load(config_path)
    for ex in load_data(cfg):
        print(ex)

if __name__ == "__main__":
    data_test("configs/gsm8k.yaml")
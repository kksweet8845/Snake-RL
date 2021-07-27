import torch
from models import PolicyGradientExecutor
import glob



if __name__ == "__main__":


    root_dir = "/home/nober/repo/MLGame/model_src/exp-0"

    pge = PolicyGradientExecutor((1, 300, 300), None, device='cpu', save_dir="./model_src", offline=True)
    for filepath in sorted(glob.glob(f"{root_dir}/*.pk")):
        print(f" \n\n========== Training {filepath} =============\n\n")
        pge.offline_training_step(filepath)

    pge.save_ckpt()




    




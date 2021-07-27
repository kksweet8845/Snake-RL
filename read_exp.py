from mlgame.env import GameRLModeExecutor
import glob
import os, pickle
from collections import deque
from games.snake.game.snake import Snake



if __name__ == "__main__":


    src = "./golden_exp"
    dst = "./golden_memory"

    os.makedirs(dst, exist_ok=True)
    global_m = deque([], maxlen=200000)
    for i, filepath in enumerate(glob.glob(f"{src}/*.pickle")):
        print(filepath)
        m = GameRLModeExecutor.read_exp(filepath, None)
        global_m.extend(m)

    with open(f"{dst}/gloden_memory.pk", 'wb') as file:
        pickle.dump(global_m, file)
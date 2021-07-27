from types import AsyncGeneratorType
from mlgame.process_manager import MultiEnvProcessManager
from mlgame.env import GameRLModeExecutor
from r_learning.models.dqn_agent import DQN_Agent
from games.snake.game.snake import Snake





if __name__ == '__main__':

    NUM_ENVS            = 16
    EXECUTOR_CLASS      = GameRLModeExecutor
    GAME_CLASS          =  Snake
    DEVICE              = 'cuda'
    AGENT               = DQN_Agent('dqn_multi_v1', batch_size=128, pretrained_src=None, device=DEVICE)

    envs_manager        = MultiEnvProcessManager(NUM_ENVS, EXECUTOR_CLASS, GAME_CLASS, device=DEVICE)
    envs_manager.set_agent(AGENT)
    envs_manager.start()


    




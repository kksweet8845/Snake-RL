import numpy as np
import time
import math
import sys
import pickle
import traceback
from collections import deque, namedtuple
import copy, os
import torch
import gym
from gym import spaces
from .wrapper import wrap_env


Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))




class GameRLModeExecutor:
    """
        The RL mode executor, which will create a environment which is used to 
        communicate with game core. It uses the pipe to communicate each other.
        And, the environment is wrapped by the 'wrap_env', which is a wrap fun-
        ction with two extended environment, resized env and stacked env.
    """
    # Set this in SOME subclasses
    metadata = {'render.modes': ['human', 'rgb_array']}
    reward_range = (-float('inf'), float('inf'))


    def __init__(self, game_cls, recv, send, game_params={}, fps=500):
        """
            Initialize the pipe and env with None.
        """
        self._game_cls = game_cls
        self.game_params = game_params
        self.recv_end = recv
        self.send_end = send
        self.env = None
        self.fps = fps
        self._rl_execution_time = 1 / self.fps
        self.frame_count = 0

    def start(self):
        # Construct game environment
        self.game = self._game_cls(**self.game_params)
        env = BaseEnv(self.game)
        self.env = wrap_env(env)
        self.frame_count = 0

        try:
            self._loop()
        except Exception as err:
            traceback.print_tb(err.__traceback__)

    def _loop(self):

        self._wait_rl_ready()
        id = os.getpid()
        done = False
        reward = 0
        info = None
        observation = self.env.reset()
        try:
            while True:
                self.send_scene_info(observation, reward, done, info)
                time.sleep(self._rl_execution_time)
                action = self.recv_end.recv()

                try:
                    observation, reward, done, info = self.env.step(action)
                except Exception as err:
                    print(f"{id} Exception occurred")
                    print(action)
                if done:
                    self.send_scene_info(observation, reward, done, info)
                    observation = self.env.reset()
                    done = False
                    reward = 0
                    info = None
                    if self._wait_rl_ready() == "QUIT":
                        break
        except Exception as err:
            traceback.print_exception(*sys.exc_info())

    def send_scene_info(self, observation, reward, done, info):
        self.send_end.send((
            observation,
            reward,
            done,
            info
        ))
            
    def _wait_rl_ready(self):
        t = self.recv_end.recv()
        while t != "READY" and t != "QUIT":
            t = self.recv_end.recv()
        return t
    

    def result(self):
        return self.game.get_game_result()

    

class BaseEnv(gym.Env):
    # Set this in SOME subclasses
    metadata = {'render.modes': ['human', 'rgb_array']}
    reward_range = (-float('inf'), float('inf'))

    def __init__(self, game, fps=30):
        self.game = game
        self.cmd_map = ["UP", "DOWN", "LEFT", "RIGHT", "NONE"]
        self.game_over = ["GAME_OVER", "RESET", "QUIT"]
        self.frame_count = 0
        self.step_left = 20
        self.apple_eaten = 0
        self.durition_time = 0
        self.reward_gamma = 1.0
        self.fdis = 100

        self.action_space = spaces.Discrete(4)
        self.observation_space = None

    
    def step(self, action):
        """ Return the observation, reward, done, info """

        cmd_map = ["UP", "DOWN", "LEFT", "RIGHT", "NONE"]
        result, fdis = self.game.update(
            {'ml' : cmd_map[action]})
        self.step_left -= 1
        self.frame_count += 1
        reward, done = self.reward_fn(result, fdis)
        
        observation = self.render(None)
        info = self.game.get_player_scene_info()['ml']

        return observation, reward, done, info

    def reward_fn(self, result, fdis):

        done = False
        game_over = ["GAME_OVER", "RESET", "QUIT"]
        reward = 0

        if self.step_left == 0:
            result = "GAME_OVER"

        if result == "GAME_ALIVE":
            reward = 0
            # self.reward_gamma *= 0.8
        elif result == "GAME_ATE_APPLE":
            reward = 10 * (1 + self.apple_eaten)
            self.apple_eaten += 1
            self.step_left += 20
            # self.reward_gamma = 1.0
        elif result in game_over:
            # reward = -10 + self.frame_count * 0.001
            reward = -10 * (1  - self.apple_eaten * 0.1)
            # self.reward_gamma = 1.0
            done = True

        # reward *= self.reward_gamma

        return reward ,done

    def reset(self):
        """ Rest the environment """
        self.game.reset()
        self.frame_count = 0
        self.step_left = 100
        self.apple_eaten = 0

        return self.render(None)

    def render(self, mode):
        """ Render the game scene """
        try:
            _2darray = self.game.get_screen()
            # _2darray = np.transpose(_2darray, (2, 0, 1))
            return _2darray

        except Exception:
            traceback.print_exception(*sys.exc_info())

    def close(self):
        pass

    def seed(self, seed=None):
        pass
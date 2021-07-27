"""
The template of the script for playing the game in the ml mode
"""

import importlib.util
import torch
import math
import numpy as np

spec = importlib.util.spec_from_file_location('dqn', '/home/nober/repo/r-learning/DQN_trainer.py')
dqn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dqn)



def reward_fn(state, prv_reward):

    head = state['snake_head']
    food = state['food']

    dis = (head[0] - food[0])**2 + (head[1]-food[1])**2
    dis = math.pow(dis, 0.5)/300

    return 1 if prv_reward != None and dis < prv_reward else -1, dis


    



class MLPlay:
    def __init__(self):
        """
        Constructor
        """
        input_dim = (1, 300, 300)
        root_dir = "/home/nober/repo/r-learning/model_src"
        self.executor = policy_gradient.PolicyGradientExecutor(input_dim, reward_fn, device='cuda', save_dir=root_dir, offline=False)
        self.t = 0
        self.epoch = 0

        self.executor = dqn.DQNEexcutor()

    def update(self, scene_info):
        """
        Generate the command according to the received scene information
        """
        if scene_info["status"] == "GAME_OVER":
            self.t += 1
            return "RESET"

        pixel = scene_info['pixel']

        tensor_pixel = torch.from_numpy(pixel)
        tensor_pixel = torch.unsqueeze(tensor_pixel, dim=0)
        tensor_pixel = tensor_pixel.to('cuda')


        action = self.executor.step(tensor_pixel, {
            'snake_head' : scene_info['snake_head'],
            'food'       : scene_info['food'],
            'snake_body' : scene_info['snake_body'],
            'status'     : scene_info['status']
        })

        del tensor_pixel

        # rand = np.random.choice(2, 1, p=[0.4, 0.6])

        # action = _action if rand[0] == 1 else np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]


        if action == 0: # up
            return "UP"
        elif action == 1: # down
            return "DOWN"
        elif action == 2: # left
            return "LEFT"
        elif action == 3: # right
            return "RIGHT"

    def reset(self):
        """
        Reset the status if needed
        """
        
        if self.t == 20:
            self.executor.save_data()
            self.t = 0
            self.epoch += 1
            print(self.epoch)

        if self.epoch == 10000:
            exit()

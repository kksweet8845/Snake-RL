import torch.nn as nn
from collections import deque
from math import pow
import torch
import logging as log
import pickle
import json
import os
# from .dataset import ExpDataset
from torch.utils.data import DataLoader
import numpy as np
import importlib.util


spec = importlib.util.spec_from_file_location('dataset', '/home/nober/repo/r-learning/models/dataset.py')
dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_module)


def SAME_PADDING(k):
    return (k-1) >> 1





class PolicyGradientExecutor:
    def __init__(self, input_dim, reward_fn, device='cpu', save_dir="./model_src", offline=False):
        
        self.agent = PolicyGradientAgent(reward_fn)
        self.model = CNNClassifier(input_dim)
        self.init = True
        self.save_dir = save_dir
        self.config = None

        with open(f"{self.save_dir}/config.json", "r") as file:
            self.config = json.load(file)
        self.org_expid = self.config.get('expid', -1)
        self.expid = self.config.get('expid', -1) + 1
        self.epoch = 0

        try:
            self.model.load_state_dict(torch.load(f"{save_dir}/saved_model-{self.org_expid}.pt"))
        except:
            pass

        if offline == True:
            # self.model.load_state_dict(torch.load(f"{save_dir}/saved_model-{self.config['expid']}.pt"))
            pass
        else:
            self.model.eval()

        self.opt = torch.optim.Adam(self.model.parameters())
        self.model.to(device)


    def reset(self):
        self.agent.reset()
        self.init = True

    def save_ckpt(self):
        """ Save the model """
        torch.save(self.model.state_dict(), f"{self.save_dir}/saved_model-{self.expid}.pt")
        with open(f"{self.save_dir}/config.json", "w") as file:
            self.config['expid'] = self.expid
            json.dump(self.config, file)


    def step(self, state, scene_info):

        out = self.model(state)

        action = torch.argmax(out)
        # print(f"prediction : {action.tolist()}")

        self.agent.record_action(out.max())
        self.agent.record_state(scene_info)
        self.agent.record_reward(scene_info)

        del out

        return action

    def mini_batch(self, samples):


        batch_size = len(samples)

        scene = np.zeros((batch_size, 1, 300, 300), dtype=np.float32)

        

        action_ls = []
        reward_ls = []
        for i, sample in enumerate(samples):
            state = sample['state']
            action = sample['action']
            reward = sample['reward']

            sh = state['snake_head']
            scene[i, 0, sh[0], sh[1]] = 3

            # snake body
            for sb in state['snake_body']:
                scene[i, 0, sb[0], sb[1]] = 2

            # snake food
            food = state['food']
            scene[i, 0, food[0], food[1]] = 1

            action_ls.append(action)
            reward_ls.append(reward)


        scene_tensor = torch.from_numpy(scene)


        return scene_tensor, action_ls, reward_ls


    def offline_training_step(self, input_file):


        dataset = dataset_module.ExpDataset(input_file)

        dataLoader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=self.mini_batch
        )

        self.klLoss = nn.KLDivLoss()
        self.beta = 0.8


        for i, (scene_tensor, action_ls, reward_ls) in enumerate(dataLoader):


            off_actions = self.model(scene_tensor)
            off_actions, idx = torch.max(off_actions, dim=1)

            loss, kl = self.offline_pgloss(off_actions, action_ls, reward_ls)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            print(f"Loss : {loss.item()}, kl: {kl}")

            del loss
            del kl



    def offline_pgloss(self, off_actions, actions, rewards):

        loss = torch.zeros(1, dtype=torch.float32)
        for off_action, action, reward in zip(off_actions, actions, rewards):
            loss = loss + reward * off_action / action

        
        kl = self.klLoss(off_actions, torch.Tensor(actions))
        loss = loss -  self.beta * kl

        loss = - loss

        return loss, kl





    def pgloss(self):
        """ The loss function """

        rewards = self.agent.rewards
        actions = self.agent.action_set

        loss = torch.zeros(1, dtype=torch.float32).to('cuda')
        for reward, action in zip(rewards, actions):
            loss = loss - reward.reward * action.action_p.log()

        print(f"loss : {loss}")
        return loss

    def update(self):

        loss = self.pgloss()

        self.opt.zero_grad()
        loss.backward(retain_graph=False)
        self.opt.step()
        self.reset()

    def save_data(self):
        
        exp_path = f"{self.save_dir}/exp-{self.expid}"
        os.makedirs(exp_path, exist_ok=True)
        exp_path = f"{exp_path}/{self.epoch}"
        self.agent.save_data(exp_path)
        self.epoch += 1
        self.reset()


class PolicyGradientAgent:
    def __init__(self, reward_fn, gama=0.7):
        """
            return the reward
        """

        self.states = []
        self.st = 0
        self.rewards = []
        self.rt = 0
        self.action_set = []
        self.at = 0
        self.rfn = reward_fn
        self.gama = gama
        self.prv_item = None
    
    def record_state(self, x):
        self.states.append(x)
        self.st += 1

    def record_reward(self, x):
        """
            Update the reward
        """

        r, dis = self.rfn(x, self.prv_item.dis if self.prv_item != None else None)

        t = RewardItem(self.rt, dis, 0)
        if self.prv_item != None:
            self.prv_item.reward = r

        self.rewards.append(t)
        self.rt += 1
        self.prv_item = t

    def record_action(self, act_p):
        self.action_set.append(ActionItem(self.at, act_p))
        self.at += 1

    def save_data(self, name):

        self.prv_item.r = -1
        table = []
        for s, a, r in zip(self.states, self.action_set, self.rewards):
            table.append({
                'state' : s,
                'action' : a.action_p.item(),
                'reward' : r.reward
            })

        with open(f'{name}.pk', 'wb') as file:
            pickle.dump(table, file)

        del table
        # with open(f'{name}.txt', 'w') as file:
        #     json.dump(table, file, separators=(':', ','), indent=4, sort_keys=True)

            
    def reset(self):
        self.rewards.clear()
        self.action_set.clear()
        for action in self.action_set:
            del action
        self.states.clear()
        self.st = 0
        self.rt = 0
        self.at = 0



        

class RewardItem:
    def __init__(self, t, dis, reward):
        self.t = t
        # self.reward = torch.tensor(reward, dtype=torch.float32)
        self.dis = dis
        self.reward = reward

class ActionItem:
    def __init__(self, t, action_p):
        self.t = t
        self.action_p = action_p


        
class Print(nn.Module):
    def __init__(self, name):
        super(Print, self).__init__()
        self.name = name
    def forward(self, x):
        print(f"[{self.name}] {x.shape}")
        return x



class CNNClassifier(nn.Module):
    def __init__(self, input_dim):
        super(CNNClassifier, self).__init__()

        c, h, w = input_dim
        self.main = nn.Sequential(
            # Conv1
            # Print('Conv0'),
            nn.Conv2d(c, c * 4, 3, stride=1, padding=SAME_PADDING(3)),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            # Print('Conv1'),
            # Conv2
            nn.Conv2d(c*4, c*8, 3, stride=1, padding=SAME_PADDING(3)),
            nn.BatchNorm2d(c*8),
            nn.ReLU(True),
            # Conv3
            nn.Conv2d(c*8, c*16, 3, stride=1, padding=SAME_PADDING(3)),
            nn.BatchNorm2d(c*16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            # Print('MaxPool2d'),
            # Flatten
            nn.Flatten(),
            # Print('Flatten'),
            # Linear1
            nn.Linear((h*w), 512),
            nn.ReLU(True),
            # nn.Dropout(p=0.5, inplace=True),
            # Print('L1'),
            # # Linear3
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            # # Linear4
            # nn.Linear(256, 32),
            # nn.ReLU(True),
            # Linear5
            nn.Linear(256, 4),
            nn.ReLU(True),
            # nn.Dropout(p=0.2, inplace=False),
            # nn.ReLU(True),
            # Softwax
            # nn.Softmax(dim=1)
            nn.Sigmoid()
        )


    def forward(self, x):
        """
            @x: the tensor of 2d array, such as Image
        """
        return self.main(x)








        
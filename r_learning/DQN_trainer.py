print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
from .models.dqn import DQN_DNN , DQN
from .models.utils.replay_memory import ReplayMemory, Transition, MeanMetrics
import random
import math
import time
import pickle

import torch
import torch.optim as optim
import torch.nn as nn
import os
from tqdm import tqdm
import traceback
import sys
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Process, Pipe
from collections import deque
import numpy as np
from .rule_base import SnakeRuleBase


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)




class RLTrainer:
    def __init__(self, env_entry_point, game_cls, input_shape, num_actions, model_src, comment='', batch_size=128, gamma=0.99,
                    eps_start=0.9, eps_end=0.05, eps_decay=200,
                    pretrained_src=None, device='cpu', num_episode=350, target_update=10):
        self.device = device
        self.model_src = model_src
        recv_for_rl, send_for_env = Pipe(False)
        recv_for_env, send_for_rl = Pipe(False)

        self.executor = DQNExecutor(input_shape, num_actions, model_src, pretrained_src=pretrained_src, device=device, batch_size=32)
        self.executor.add_pipe(recv_for_rl, send_for_rl)

        self.env_process = Process(target = env_entry_point,
                                name="env process", args=(game_cls, recv_for_env, send_for_env))

        self.env_process.start()

        self.fitness_record = deque([], maxlen=3000)
        self.num_episode = num_episode
        self.action_record = np.zeros(5, dtype=float)
        self.TARGET_UPDATE = target_update

        self.executor.load_memory()
        self.avg = 100

        
    def gen_sample_data(self, n):
        
        init_model = True
        chance = 100
        avg = 600

        os.makedirs(f"{self.model_src}/gen_model_0", exist_ok=True)
        while len(self.fitness_record) <= 800:
            pbar = tqdm(range(n))
            for i in pbar:
                rid = int(random.random() * int(1e15))
                fitness, sample_data = self.gen_network_from_scratch(rid, avg, init_model=init_model, save_model=False)
                pbar.set_postfix({"Data" : sample_data, 'Fitness': fitness })
                self.executor.writer.add_scalar("Fitness", fitness, i)

        with open(f'{self.model_src}/gen_model_0/fitness.pk', 'wb') as file:
            pickle.dump(self.fitness_record, file)
        self.executor.dump_memory()
    

    def gen_2000_sample_data(self):

        pbar = tqdm(range(2000))
        for i in pbar:
            fitness, sample_data = self.gen_network_from_scratch(i, self.avg)
            pbar.set_postfix({"Data" : sample_data, 'Fitness': fitness })

        with open(f'{self.model_src}/gen_model_0/fitness.pk', 'wb') as file:
            pickle.dump(self.fitness_record, file)

    def gen_network_from_scratch(self, i, avg, save_model=True, init_model=True):
        """ It will collect sample data in here """


        def np2tensor(arr):
            t = torch.from_numpy(arr)
            t = t.squeeze(dim=1)
            return t

        frame_count = 0

        if init_model:
            self.executor.simply_init_model()
        # Executor ready
        self.executor.send("READY")
        info = self.executor.recv()
        cur_state = info['pixel']
        cur_state = np2tensor(cur_state)
        next_state = None
        env_info = None

        loss = 0
        DEVICE = self.device
        
        fitness = 0
        while True:
            # Response
            action = self.executor.select_action_by_rule(info)
            # action = self.executor.select_action(cur_state)
            print(f"action: {action}")
            self.executor.send(action.item())
            frame_count += 1

            # Wait for next state
            info = self.executor.wait_until_response()

            next_state, reward, done, score = np2tensor(info['pixel']),  \
                                              torch.tensor([info['reward']], device=DEVICE), \
                                              info['done'], \
                                              info['score']
            
            agent_frame = info['frame']

            self.executor.push(cur_state, action, next_state, reward)

            cur_state = next_state
            fitness += reward.item()
            if done:
                print("Die")
                if fitness > avg:
                    self.executor.push_to_memory()
                else:
                    self.executor.clear_buf_mem()
                break
            frame_count += 1
            delay_frame = agent_frame - frame_count
            if delay_frame > 0:
                print(f"delay {delay_frame}")

        # fitness = math.floor(frame_count * frame_count) * math.pow(2, score)


        if save_model and fitness >= avg:
            print(f"Save {self.model_src}/gen_model_0/{i}")
            self.fitness_record.append(
                ( f"{self.model_src}/gen_model_0/{i}" , fitness)
            )
        
            # Save model
            self.executor.save_model(f"{self.model_src}/gen_model_0/{i}")

        return fitness, len(self.executor.memory)
    
    def play_1_episode_with_optimize(self, pbar, avg, episode):

        def np2tensor(arr):
            t = torch.from_numpy(arr)
            t = t.squeeze(dim=1)
            return t

        frame_count = 0

        # Executor ready
        self.executor.send("READY")
        cur_state = self.executor.recv()['pixel']
        cur_state = np2tensor(cur_state)
        next_state = None
        env_info = None

        loss = 0
        DEVICE = self.device
        
        fitness = 0

        while True:
            # Response
            action = self.executor.select_action(cur_state)
            # action = self.executor.select_action_by_model(cur_state)
            self.executor.send(action.item())
            frame_count += 1

            # Wait for next state
            info = self.executor.wait_until_response()

            next_state, reward, done, score = np2tensor(info['pixel']),  \
                                              torch.tensor([info['reward']], device=DEVICE), \
                                              info['done'], \
                                              info['score']
            
            agent_frame = info['frame']

            self.executor.push(cur_state, action, next_state, reward)

            cur_state = next_state
            fitness += reward.item()
            loss = self.executor.optimize_model()
            self.action_record[action.item()] += 1
            if done:
                # it will check if it is above the average
                if fitness > avg:
                    self.executor.push_to_memory()
                self.executor.record(score, fitness, episode, eval=False)
                norm = np.linalg.norm(self.action_record)
                pbar.set_postfix({'loss' : loss, 'actions': self.action_record/norm, 'sample_data' : f"{len(self.executor.memory):e}"})
                frame_count = 0
                break

            delay_frame = agent_frame - frame_count
            if delay_frame > 0:
                print(f"delay {delay_frame}")
            
            frame_count += 1

    def play_1_episode_for_eval(self, episode):
        """ Default the model is loaded """
        def np2tensor(arr):
            t = torch.from_numpy(arr)
            t = t.squeeze(dim=1)
            return t

        frame_count = 0

        # Executor ready
        self.executor.send("READY")
        cur_state = self.executor.recv()['pixel']
        cur_state = np2tensor(cur_state)
        next_state = None
        env_info = None

        loss = 0
        DEVICE = self.device
        
        fitness = 0
        while True:
            # Response
            action = self.executor.select_action_by_model(cur_state)
            self.executor.send(action.item())
            frame_count += 1

            # Wait for next state
            info = self.executor.wait_until_response()

            next_state, reward, done, score = np2tensor(info['pixel']),  \
                                              torch.tensor([info['reward']], device=DEVICE), \
                                              info['done'], \
                                              info['score']
            
            agent_frame = info['frame']

            cur_state = next_state
            fitness += reward.item()
            if done:
                self.executor.record(score, fitness, episode, eval=True)
                break

            delay_frame = agent_frame - frame_count
            if delay_frame > 0:
                print(f"delay {delay_frame}")

            
            frame_count += 1
        
        return fitness, score
        # if score < 10:
        #     return math.floor(frame_count * frame_count) * math.pow(2, score), score
        # else:
        #     return math.floor(frame_count * frame_count ) * math.pow(2, 10) * (score - 9), score

    def select_top_n_weights(self, n, from_disk=True):

        try:

            if from_disk:
                with open(f"{self.model_src}/gen_model_0/fitness.pk", 'rb') as file:
                    self.fitness_record = pickle.load(file)
            
            # print(self.fitness_record)
            dtype = [('source', 'S100'), ('fitness', float)]
            arr = np.array(self.fitness_record, dtype=dtype)
            
            arr = np.sort(arr, order='fitness')

            arr = arr[-n:]
            print(arr)
            return arr
        except Exception as err:
            pass

    def select_top_n_weights_from_path(self, n, generation : int):

        with open(f"{self.model_src}/gen_model_{generation}/fitness.pk", 'rb') as file:
                self.fitness_record = pickle.load(file)

        dtype = [('source', 'S100'), ('fitness', float)]
        arr = np.array(self.fitness_record, dtype=dtype)
        
        arr = np.sort(arr, order='fitness')

        arr = arr[-n:]
        print(arr)
        return arr
        

    def training_one_model(self, model_record):

        pbar = tqdm(range(self.num_episode))
        train_bar = tqdm(range(int(1e5)))
        # wbpath, fitness = model_record
        # wpath = wbpath.decode('utf-8')
        # id = wpath.split('/')[-1]
        pbar = tqdm(range(self.num_episode))
        # self.executor.load_model(wpath)

        self.executor.open_SummaryWriter(log_dir=None)
        self.action_record = [0] * 4

        for i in train_bar:
            loss = self.executor.optimize_model()
            self.executor.writer.add_scalar("Eval/Loss", loss, i)
            train_bar.set_postfix({"Loss": loss})
            if i % 10:
                self.executor.update_network()

        self.executor.update_network()

        for i_episode in pbar:
            self.play_1_episode_with_optimize(pbar, self.avg, i_episode)

            if i_episode % 10 == 0:
                self.executor.update_network()

        new_fitness = 0
        avg_score = 0
        for i in range(10):
            tmp_fitness, score = self.play_1_episode_for_eval(i)
            new_fitness += tmp_fitness
            avg_score += score

        self.executor.close_SummaryWriter()
        
        new_fitness /= 10
        avg_score /= 10
        print(f"Fitness: {new_fitness}, score: {avg_score}")


        

    def training(self, candicate_model_weights, generation : int):

        new_candicate_network = deque([], maxlen=3000)
        next_generation = generation + 1
        os.makedirs(f"{self.model_src}/gen_model_{next_generation}/", exist_ok=True)
        
        for wbpath, fitness in candicate_model_weights:
            wpath = wbpath.decode('utf-8')
            id = wpath.split('/')[-1]
            pbar = tqdm(range(self.num_episode))
            self.executor.load_model(wpath)

            self.executor.open_SummaryWriter(log_dir=os.path.join("./runs", wpath))
            self.action_record = [0] * 4
            for i_episode in pbar:
                self.play_1_episode_with_optimize(pbar, self.avg, i_episode)

                if i_episode % 10 == 0:
                    self.executor.update_network()

            new_fitness = 0
            avg_score = 0
            for i in range(10):
                tmp_fitness, score = self.play_1_episode_for_eval(i)
                new_fitness += tmp_fitness
                avg_score += score

            self.executor.close_SummaryWriter()
            
            new_fitness /= 10
            avg_score /= 10
            print(f"Fitness : {fitness}, trained: {new_fitness}, score: {avg_score}")
            if fitness < new_fitness:
                print("Select as next generation")
                self.executor.save_model(f"{self.model_src}/gen_model_{next_generation}/{id}")
                new_candicate_network.append((f"{self.model_src}/gen_model_{next_generation}/{id}", new_fitness))

        self.executor.dump_memory()
        self.executor.memory.discard(0.8)
        if len(new_candicate_network) == 0:
            return None

        with open(f"{self.model_src}/gen_model_{next_generation}/fitness.pk", "wb") as file:
            pickle.dump(new_candicate_network, file)

        return new_candicate_network

    def evaluate_model(self, path):

        with open(f"{path}/fitness.pk") as file:
            fitness_record = pickle.load(file)

        for wpath, fitness in fitness_record:
            self.executor.load_model(wpath)
            self.play_1_episode_for_eval()
    
    def evaluate_one_model(self, path):

        self.executor.load_model(path)
        fitness = self.play_1_episode_for_eval()
        return fitness

    def dump_memory(self):
        self.executor.dump_memory()

    def path(self, generation : int):
        return f"{self.model_src}/gen_model_{generation}"

    def close_env(self):
        self.executor.send("QUIT")

    def gen_network_based_on_target_net(self):
        pass





class DQNExecutor:
    def __init__(self, input_shape, num_actions, model_src, comment='', batch_size=128,
                gamma=0.999, eps_start=0.9, eps_end=0.05,
                eps_decay=200, pretrained_src=None, device='cpu'):

        self.model_src = model_src
        self.pretrained_src = pretrained_src
        self.screen_height, self.screen_width = input_shape
        self.num_actions = num_actions
        self.device = device
        self.batch_size = batch_size
        # self.policy_net = DQN_DNN(24, self.num_actions, device=device).to(device)
        # self.target_net = DQN_DNN(24, self.num_actions, device=device).to(device)
        self.policy_net = DQN(self.screen_height, self.screen_width, self.num_actions, device=device).to(device)
        self.target_net = DQN(self.screen_height, self.screen_width, self.num_actions, device=device).to(device)
        self.training_step = 0
        if pretrained_src != None:
            print(f"Load from {self.pretrained_src}")
            self.policy_net.load_state_dict(torch.load(f"{self.pretrained_src}/policy_net.pb"))
            self.target_net.load_state_dict(torch.load(f"{self.pretrained_src}/target_net.pb"))
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Two net equal
        
        self.target_net.eval()
        self.opt = optim.RMSprop(self.policy_net.parameters(), lr=1e-5)
        self.memory = ReplayMemory(int(100000), self.model_src)
        self.steps_done = 0
        self.EPS_START = eps_start
        self.EPS_END   = eps_end
        self.EPS_DECAY = eps_decay
        self.GAMMA = gamma
        self.EPS_DEGRADING = 10
        self.avg_loss = MeanMetrics()
        self.temp_reward = 0
        
        # os.makedirs(f"{self.model_src}/", exist_ok=True)
        self.writer = SummaryWriter(comment=comment)

        # self.snakeRule = SnakeRuleBase((300, 300))

    def open_SummaryWriter(self, log_dir=None):
        self.writer.close()
        self.writer = SummaryWriter(log_dir=log_dir)

    def close_SummaryWriter(self):
        self.writer.close()

    def add_pipe(self, recv, send):
        self.recv_end = recv
        self.send_end = send

    def send(self, x):
        self.send_end.send(x)

    def recv(self):
        return self.recv_end.recv()

    def wait_until_response(self):
        
        while True:
            t = self.recv_end.recv()
            if t != None:
                return t

    def load_from_two_model(self, path1, path2, ratio):

        self.policy_net.load_state_dict(torch.load(f"{path1}/policy_net.pb"))
        self.target_net.load_state_dict(torch.load(f"{path2}/policy_net.pb"))


        model1 = self.policy_net
        model2 = self.target_net

        with torch.no_grad():
            for layer1, layer2 in zip(model1.layers, model2.layers):
                for pt1 in layer1.state_dict():
                    tensor1 = layer1.state_dict()[pt1]
                    tensor2 = layer2.state_dict()[pt1]
                    mask = torch.empty(tensor1.size()).to(self.device)
                    mask = torch.nn.init.uniform_(mask)
                    mask = mask.masked_fill(mask < ratio, 0)
                    tensor1 = tensor1.masked_fill(mask == 0, 0)
                    tensor2 = tensor2.masked_fill(mask != 0, 0)
                    tensor1 += tensor2
                    tensor1 = torch.clamp(tensor1, -1, 1)

    def simply_init_model(self):
        self.policy_net.init_weights()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.steps_done = 0

    def reinit_model_without_check(self, ratio):
        self.policy_net.randomly_alter(ratio)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def init_model_without_check(self):
        self.policy_net.init_weights()
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def record(self, score, reward, i, eval=False):
        if not eval:
            self.writer.add_scalar("Score", score, i)
            self.writer.add_scalar("Reward", reward, i)
        else:
            self.writer.add_scalar("Eval/Score", score, i)
            self.writer.add_scalar("Eval/Reward", reward, i)


    def push(self, t, a, nt, r):
        self.memory.push(t, a, nt, r)

    def push_to_memory(self):
        self.memory.push_to_memory()

    def dump_memory(self):
        self.memory.save(self.model_src)

    def load_memory(self):
        self.memory.load(self.model_src)

    def clear_buf_mem(self):
        self.memory.reset_buf_mem()

    def select_action(self, state):
        sample = random.random()


        eps_threshold = self.EPS_END + ( self.EPS_START - self.EPS_END ) * \
                            math.exp(-1. * self.steps_done / self.EPS_DECAY)
    
        self.steps_done += 1
        try:
            if sample > eps_threshold:
                with torch.no_grad():
                    a = self.policy_net(state).max(1)[1].view(1, 1)
                    return a
            else:
                return torch.tensor([[random.randrange(self.num_actions)]], device=self.device,
                    dtype=torch.long)
        except Exception as err:
            traceback.print_exception(*sys.exc_info())
    
    def select_action_by_model(self, state):
        return self.policy_net(state).max(1)[1].view(1, 1)

    def select_action_by_rule(self, info):

        
        eps_threshold = self.EPS_END + ( self.EPS_START - self.EPS_END ) * \
                            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        # if random.random() >  eps_threshold:
        #     return torch.tensor([[random.randrange(self.num_actions)]], device=self.device,
        #             dtype=torch.long)

        return self.snakeRule.select_action_by_rule(info)

        # if hx > fx:
        #     return torch.tensor([[2]], device=self.device,
        #             dtype=torch.long)
        # elif hx < fx:
        #     return torch.tensor([[3]], device=self.device,
        #             dtype=torch.long)
        # elif hy > fy:
        #     return torch.tensor([[0]], device=self.device,
        #             dtype=torch.long)
        # elif hy < fy:
        #     return torch.tensor([[1]], device=self.device,
        #             dtype=torch.long)


    def train_step(self):

        BATCH_SIZE = self.batch_size
        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device,
                                                dtype=torch.bool)

        
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch


        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.opt.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.opt.step()

        return loss.item()
    
    def optimize_model(self):
        if self.memory.bad_reward_samples_len() < (self.batch_size // 2) or len(self.memory.good_reward_memory) <  (self.batch_size // 2):
            return 0

        # print(f"Bad sample data : {self.memory.bad_reward_samples_len()}")
        mem_len = len(self.memory)
        BATCH_SIZE = self.batch_size
        loss = 0
        # for i in tqdm(range(int(mem_len / BATCH_SIZE))):
        #     loss += self.train_step()
        # for i in range(2):
        for i in range(1):
            loss = self.train_step()
            self.writer.add_scalar("Loss", loss, self.training_step)
            self.training_step += 1
            self.avg_loss.update(loss)
        # for i in range(30):
            # loss += self.train_step()

        return self.avg_loss.mean()

        return loss / (mem_len/BATCH_SIZE)
        

    def update_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # self.steps_done = 0

    def load_model(self, src):
        print(f"Load from {src}")
        self.policy_net.load_state_dict(torch.load(f"{src}/policy_net.pb"))
        self.target_net.load_state_dict(torch.load(f"{src}/target_net.pb"))

    def save_model_directly(self):
        os.makedirs(self.model_src, exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"{self.model_src}/policy_net.pb")
        torch.save(self.target_net.state_dict(), f"{self.model_src}/target_net.pb")

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"{path}/policy_net.pb")
        torch.save(self.target_net.state_dict(), f"{path}/target_net.pb")
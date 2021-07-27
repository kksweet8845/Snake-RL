from multiprocessing import Process, ProcessError, Pipe
from multiprocessing.pool import Pool
import sys, os
import traceback
import torch
from collections import deque
import numpy as np


def entry_point(executor_cls, recv, send, game_cls, game_params):
    env_process = _MultiEnvProcess((recv, send), executor_cls, game_cls, game_params)
    env_process.start()

def np2tensor(arr):

    t = torch.from_numpy(arr)
    t = torch.unsqueeze(t, dim=0)
    return t

def wait_for_env(i, recv):
    a = 0
    while True:
        t = recv.recv()
        if t != None:
            return i, t



class MeanMetrics:
    def __init__(self, x=0):
        self.avg = 0
        self.i = 0
        self.avg_diff = 0
        self.avg_lv = 1
        self.i_v = 0
        self.i_d = 0
        self.slow_diff = 0
        self.ls_10 = deque([], maxlen=10)
        self.ls_50 = deque([], maxlen=50)
        self.ls_100 = deque([], maxlen=100)
        self.cur_x = 0
        self.best = None

    def update(self, x):

        new_record = False

        if self.best == None:
            self.best = x
            new_record = True
        elif self.best < x:
            self.best = x
            new_record = True

        avg = (self.avg * self.i + x) / (self.i + 1)
        self.i += 1
        self.ls_10.append(x)
        self.ls_50.append(x)
        self.ls_100.append(x)
        self.avg = avg

        return new_record

    def mean(self):
        return self.avg

    def reset(self):
        self.avg = 0
        self.avg_diff = 0
        self.avg_lv = 1
        self.i = 0

    def last_10_avg(self):
        if len(self.ls_10) >= 1:
            return np.mean(self.ls_10)
        return 10241024
    
    def last_50_avg(self):
        if len(self.ls_50) >= 1:
            return np.mean(self.ls_50)
        return 10241024 

    def last_100_avg(self):
        if len(self.ls_100) >= 1:
            return np.mean(self.ls_100)
        return 10241024




class MultiEnvProcessManager:
    """ A process manager manages multi environment (process) """

    def __init__(self, num_envs, executor_cls, game_cls, game_params={}, device='cpu'):

        self.num_envs = num_envs
        self.env_processes = None
        self.agent = None

        self.executor_cls = executor_cls
        self.game_cls = game_cls
        self.game_params = game_params
        self.pool_workers = Pool(16)
        self.device = device
        self.create_multi_env()

        self.rewards_record = [0] * self.num_envs
        self.best_reward = None
        self.avg_reward = MeanMetrics()
        self.num_games = 0

    def stats(self, rewards, dones, infoes):

        new_record = False
        for i, _ in enumerate(zip(rewards, dones, infoes)):
            reward, done, info = _
            if done:
                self.rewards_record[i] += reward.item()
                new_record = self.avg_reward.update(self.rewards_record[i])
                self.rewards_record[i] = 0
                self.num_games += 1
            else:
                self.rewards_record[i] += reward.item()
            
        if new_record:
            print(f"="*15, end='')
            print("New record", end='')
            print('='*15)
            print(f"Best reward : {self.avg_reward.best}")
        
        
        st =  f"========== {self.num_games} =========\n"
        st += f"Last 100 average reward : {self.avg_reward.last_100_avg()}\n"
        st += f"Average reward          : {self.avg_reward.mean()}"

        return st
                

            

    def set_agent(self, agent):
        self.agent = agent

    def create_multi_env(self):

        self.env_processes = []
        self.agent_recv_ends = []
        self.agent_send_ends = []
        for i in range(self.num_envs):
            env_recv, agent_send = Pipe(False)
            agent_recv, env_send = Pipe(False)
            idx = i
            process = Process(
                target = entry_point, name=f'{idx}-env',
                args=(self.executor_cls, env_recv, env_send, self.game_cls, {'id' : i}, )
            )
            self.env_processes.append(process)
            self.agent_recv_ends.append(agent_recv)
            self.agent_send_ends.append(agent_send)
    
    def start(self):
        # start the environment
        self.create_multi_env()
        for process in self.env_processes:
            process.start()

        try:
            assert self.agent != None, "Agent should not be None"
            self._loop()
        except Exception:
            traceback.print_exception(*sys.exc_info())


    def _loop(self):
        """ 
            Control the agent to communicate with multi environment.
        """
        try:
            DEVICE = self.device
            while True:

                self.send_msg_all(["READY"]*self.num_envs)
                res = self.recv_msg_all()
                cur_states, rewards, dones, infoes = tuple(zip(*res))
                cur_states = list(map(lambda x: np2tensor(x), cur_states))

                while True:

                    actions_tensor = self.agent.act_batch(cur_states)
                    actions = list(map(lambda x : x.item(), actions_tensor))
                    self.send_msg_all(actions)

                    res = self.recv_msg_all()
                    next_states, rewards, dones, infoes = tuple(zip(*res))
                    next_states = list(map(lambda x: np2tensor(x), next_states))
                    rewards = list(map(lambda x: torch.tensor([x], device=DEVICE), rewards))

                    self.agent.memorize_batch(cur_states, actions_tensor, next_states, rewards)
                    stats = self.stats(rewards, dones, infoes)

                    cur_states = next_states
                    self.agent.update()

                    if self.num_games != 0 and \
                        self.num_games % 100 == 0:
                        self.agent.update_network()
                    
                    if self.num_games != 0 and self.num_games % 500 == 0:
                        print(stats)

                    self.send_ready_by_case(dones)

        except Exception as err:
            traceback.print_exception(*sys.exc_info())

        finally:
            self.agent.save_model()
            self.send_msg_all(['QUIT'] * self.num_envs)
            print("Model saved")

    def send_ready_by_case(self, dones):
        for i, done in enumerate(dones):
            if done:
                self.agent_send_ends[i].send("READY")

    def send_msg_all(self, msgs):
        for send_end, msg in zip(self.agent_send_ends, msgs):
            send_end.send(msg)

    def recv_msg_all(self):

        async_res_ls = deque([])
        for i, recv in enumerate(self.agent_recv_ends):
            async_res = self.pool_workers.apply_async(wait_for_env, args=(i, recv, ))
            async_res_ls.append(async_res)

        
        res = [None] * self.num_envs
        try:
            for async_res in async_res_ls:
                i, r = async_res.get(1)
                res[i] = r
            # while len(async_res_ls) != 0:
            #     async_res = async_res_ls.popleft()
            #     if async_res.ready():
            #         i, r = async_res.get(1)
            #         res[i] = r
            #     else:
            #         async_res_ls.append(async_res)
        except TimeoutError as err:
            traceback.print_exception(*sys.exc_info())

        for r in res:
            assert r != None, "Return should not be None"

        return res
        

class _MultiEnvProcess:
    """
        Create multiple environment to interact with agent.
    """
    def __init__(self, pipes : tuple, executor_cls, game_cls, game_params, fps=100):

        self.recv_end, self.send_end = pipes
        self.executor = executor_cls(game_cls, self.recv_end, self.send_end, game_params=game_params, fps=fps)

    def start(self):
        self.executor.start()

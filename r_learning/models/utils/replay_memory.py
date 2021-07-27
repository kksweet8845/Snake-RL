from collections import namedtuple, deque
import random
import numpy as np
import pickle
import torch
import math
import os

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))


class MeanMetrics:
    def __init__(self, x=0):
        self.avg = x
        self.i = 0
        self.avg_diff = 0
        self.avg_lv = 1
        self.i_v = 0
        self.i_d = 0
        self.slow_diff = 0
        self.ls_10 = deque([], maxlen=10)
        self.ls_50 = deque([], maxlen=50)
        self.cur_x = 0
        self.best = None

    def update(self, x):
        if self.best == None:
            self.best = x
        elif self.best < x:
            self.best = x

        avg = (self.avg * self.i + x) / (self.i + 1)
        self.ls_10.append(x)
        self.ls_50.append(x)
        self.cur_x = x
        if self.i % 100 == 0:
            self.avg_diff = (self.avg_diff * self.i_d + (avg - self.avg)) / (self.i_d + 1)
            self.avg_lv = (self.avg_lv * self.i_v + avg/(self.avg+1e-20)) / (self.i_v + 1)
            self.i += 1
            self.i_d += 1
        self.avg = avg

    def mean(self):
        return self.avg

    def reset(self):
        self.avg = 0
        self.avg_diff = 0
        self.avg_lv = 1
        self.i = 0

    def trend(self):
        return self.avg_diff

    def level_trend(self):
        return self.avg_lv
    
    def last_10_avg(self):
        if len(self.ls_10) >= 1:
            return np.mean(self.ls_10)
        else:
            return 10241024
    
    def last_50_avg(self):
        if len(self.ls_50) >= 1:
            return np.mean(self.ls_50)
        else:
            return 10241024 


def randomly_merge_list(list1, list2):

    ls = []
    while len(list1) > 0 and len(list2) > 0:
        if random.random() > 0.5:
            t = list1.pop()
            ls.append(t)
        else:
            t = list2.pop()
            ls.append(t)

    if len(list1) > 0:
        ls.extend(list1)

    if len(list2) > 0:
        ls.extend(list2)


    return ls




class ReplayMemory(object):
    def __init__(self, capacity, model_src, exp_fitness=500):

        self.good_reward_memory = deque([], maxlen=capacity)
        self.bad_reward_meory = deque([], maxlen=capacity)
        self.good_buf = deque([], maxlen=1000)
        self.bad_buf = deque([], maxlen=1000)


        self.memory_path_list = []
        self.capacity = capacity
        self.model_src = model_src


        os.makedirs(f"{self.model_src}/good_memory", exist_ok=True)
        os.makedirs(f"{self.model_src}/bad_memory", exist_ok=True)

    def push(self, s, a, ns, r):
        """ Save a transition """

        if r > 0:
            # if len(self.good_reward_memory) + 1 > self.capacity:
            # self.good_reward_memory.append(Transition(s, a, ns, r))
            self.good_reward_memory.append(Transition(s, a, ns, r))
        else:
            self.bad_reward_meory.append(Transition(s, a, ns, r))
            # self.bad_reward_meory.append(Transition(s, a, ns, r))
        # self.memory.append(Transition(s, a, ns, r))
        # self.buf.append(Transition(s, a, ns, r))
        # self.fitness += r.item()

    def usage(self):
        return len(self.memory) / self.memory.maxlen

    def push_to_memory(self):
        self.good_reward_memory.extend(self.good_buf.copy())
        self.bad_reward_meory.extend(self.bad_buf.copy())
        self.good_buf.clear()
        self.bad_buf.clear()
        

    def push_to_memory_directly(self):
        # self.buf_avg_fitness.update(self.fitness)
        # self.memory.extend(self.buf.copy())
        pass

    def reset_buf_mem(self):
        # self.buf.clear()
        # self.fitness = 0
        self.good_buf.clear()
        self.bad_buf.clear()

    def reset(self):
        self.good_reward_memory.clear()
        self.bad_reward_meory.clear()

    def sample(self, batch_size):
        # print(math.floor(batch_size/2), len(self.bad_reward_meory))
        bad_sample = random.sample(self.bad_reward_meory, math.floor(batch_size/2))
        good_sample = random.sample(self.good_reward_memory, math.floor(batch_size/2))

        return randomly_merge_list(bad_sample, good_sample)
    
    def save(self, path):

        with open(f"{path}/good_memory.pk", "wb") as file:
            print(f"Saved {len(self.good_reward_memory)}")
            pickle.dump(self.good_reward_memory, file)
    
        with open(f"{path}/bad_memory.pk", "wb") as file:
            print(f"Saved {len(self.bad_reward_meory)}")
            pickle.dump(self.bad_reward_meory, file)
    
    def load(self, path):

        gp = f"{path}/good_memory.pk"
        bp = f"{path}/bad_memory.pk"

        if os.path.isfile(gp):
            with open(gp, "rb") as file:
                loaded = pickle.load(file)
                self.good_reward_memory.extend(loaded)
        if os.path.isfile(bp):
            with open(bp, "rb") as file:
                loaded = pickle.load(file)
                self.bad_reward_meory.extend(loaded)
                print(f"{len(self.bad_reward_meory)}")

    def bad_reward_samples_len(self):
        return len(self.bad_reward_meory)

    def discard(self):

        num_to_discard = len(self.good_reward_memory) + len(self.bad_reward_meory)

        while num_to_discard > 0:
            t = self.good_reward_memory.popleft()
            num_to_discard -= 1
            del t
        


    def __len__(self):
        return len(self.bad_reward_meory) + len(self.good_reward_memory)








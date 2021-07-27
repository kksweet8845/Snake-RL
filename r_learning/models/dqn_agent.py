from r_learning.models.utils.replay_memory import ReplayMemory, Transition, MeanMetrics
from .dqn import DQN
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


import traceback, sys, random, math, os


default_config = {
        'input_shape' : (10, 10),
        'num_actions' : 4,
        'eps_start' : 0.9,
        'eps_end' : 0.05,
        'eps_decay' : 200,
        'gamma' : 0.999
    }

# class DefaultConfig:
#     def __init__(self):
#         self.input_shape = (10, 10)
#         self.num_actions = 4,
#         self.eps_start = 0.9
#         self.eps_decay = 200
#         self.eps_end = 0.05

class DQN_Agent:
    """
        The agent will communicate with environment with
        the following steps.
        1. act
        2. memorize
        3. update
    """
    def __init__(self, model_src, config=default_config, batch_size=128, pretrained_src=None, device='cpu'):
        
        # Fundamental hyper argument steup
        self.model_src = model_src
        self.pretrained_src = pretrained_src
        self.screen_height, self.screen_width = config['input_shape']
        self.num_actions = config['num_actions']
        self.device = device
        self.batch_size = batch_size


        # Set up two nets
        self.policy_net = DQN(self.screen_height, self.screen_width, self.num_actions, device=device).to(device)
        self.target_net = DQN(self.screen_height, self.screen_width, self.num_actions, device=device).to(device)
        
        if pretrained_src != None:
            try:
                print(f"Load from {self.pretrained_src}")
                self.policy_net.load_state_dict(torch.load(f"{self.pretrained_src}/policy_net.pb"))
                self.target_net.load_state_dict(torch.load(f"{self.pretrained_src}/target_net.pb"))
            except IOError as err:
                traceback.print_exception(*sys.exc_info())
                print(f"Missing policynet.pb or target_net.pb in {self.pretrained_src}")
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Set up the parameter of agent
        self.opt = optim.RMSprop(self.policy_net.parameters(), lr=1e-5)
        self.memory = ReplayMemory(int(100000), self.model_src)
        self.steps_done = 0
        self.training_step = 0
        self.EPS_START = config['eps_start']
        self.EPS_END = config['eps_end']
        self.EPS_DECAY = config['eps_decay']
        self.GAMMA = config['gamma']
        self.EPS_DEGRADING = 10

        # Set up recorder
        self.avg_loss = MeanMetrics()

        # Summary writter
        self.writer = SummaryWriter()

    def memorize_batch(self, cur_states, actions, next_states, rewards):
        for t, a, nt, r in zip(cur_states, actions, next_states, rewards):
            self.memorize(t, a, nt, r)

    def memorize(self, t, a, nt, r):
        self.memory.push(t, a, nt, r)

    def act(self, state):
        return self._select_action(state)
    
    def act_batch(self, states):

        ls = []
        for state in states:
            ls.append(self.act(state))

        return ls

    def update(self):
        return self._optim_model()

    def update_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
            
    def _select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                    math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        try:
            if sample > eps_threshold:
                with torch.no_grad():
                    a = self.policy_net(state).max(1)[1].view(1, 1)
                    return a
            else:
                return torch.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=torch.long)
        except Exception as err:
            traceback.print_exception(*sys.exc_info())

    def _optim_model(self):
        if self.memory.bad_reward_samples_len() < (self.batch_size // 2) or len(self.memory.good_reward_memory) <  (self.batch_size // 2):
            return 0

        loss = 0
        for i in range(1):
            loss = self._train_step()
            self.writer.add_scalar("Loss", loss, self.training_step)
            self.training_step += 1
            self.avg_loss.update(loss)

        return self.avg_loss.mean()

    def _train_step(self):

        BATCH_SIZE = self.batch_size
        transitions = self.memory.sample(BATCH_SIZE)
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
            
    def open_SummaryWriter(self, log_dir=None):
        self.writer.close()
        self.writer = SummaryWriter(log_dir=log_dir)

    def close_SummaryWriter(self):
        self.writer.close()

    def load_model(self, src):
        try:
            print(f"Load from {src}")
            self.policy_net.load_state_dict(torch.load(f"{src}/policy_net.pb"))
            self.target_net.load_state_dict(torch.load(f"{src}/target_net.pb"))
        except IOError as err:
            traceback.print_exception(*sys.exc_info())
            print(f"Missing policynet.pb or target_net.pb in {src}")

    def save_model(self, dst=None):
        dst = self.model_src if dst == None else dst
        os.makedirs(dst, exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"{dst}/policy_net.pb")
        torch.save(self.target_net.state_dict(), f"{dst}/target_net.pb")

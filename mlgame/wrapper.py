import gym
from gym import spaces
import numpy as np
import cv2
from collections import deque
import traceback
import sys

class EnvWarpper(gym.Env):

    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reward_range = env.reward_range

        # self.clock = env.clock
        # self.process_events = env.process_events
        # self.should_quit = env.should_quit

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

class ResizeAndGrayscaleWrapper(EnvWarpper):
    
    def __init__(self, env, w, h):
        super(ResizeAndGrayscaleWrapper, self).__init__(env)

        self.observation_space = spaces.Box(0, 1, shape=[w, h], dtype=np.float32)
        self.w = w
        self.h = h

    def _observation(self, obs):
        try:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = cv2.resize(obs, (self.w, self.h), interpolation=cv2.INTER_AREA)
            obs = obs.astype(np.float32) / 255.0
            return obs
        except Exception: 
            traceback.print_exception(*sys.exc_info())

    def reset(self):
        return self._observation(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._observation(obs)
        return obs, reward, done, info


class StackFramesWrapper(EnvWarpper):
    def __init__(self, env, num_frames):
        super(StackFramesWrapper, self).__init__(env)
        if len(env.observation_space.shape) != 2:
            raise Exception('Stack frames works with 2D single channel images')
        self.num_frames = num_frames
        self.frames = None

        new_obs_space_shape = env.observation_space.shape + (num_frames, )
        self.observation_space = spaces.Box(0.0, 1.0, shape=new_obs_space_shape, dtype=np.float32)

    
    def _frame_as_numpy(self):
        def numpy_all_the_way(list_of_array):
            try:
                shape = list(list_of_array[0].shape)
                shape[:0] = [len(list_of_array)]
                arr = np.concatenate(list_of_array).reshape(shape)
                return arr
            except Exception:
                traceback.print_exception(*sys.exc_info())

        return numpy_all_the_way(self.frames)
    
    def reset(self):
        observation = self.env.reset()
        self.frames = deque([observation] * self.num_frames)
        # print(self.frames)
        return self._frame_as_numpy()


    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.popleft()
        self.frames.append(observation)

        return self._frame_as_numpy(), reward, done, info




def wrap_env(env, sz=10, num_frames=3):
    """ The wrapper for the 2d dynamic game"""
    env = ResizeAndGrayscaleWrapper(env, sz, sz)
    env = StackFramesWrapper(env, num_frames)
    return env





from os import wait
from mlgame.env import GameRLModeExecutor
from games.snake.game.snake import Snake
import torch
import numpy as np
import importlib.util
from r_learning import DQN_trainer as dqn
from tqdm import tqdm
from multiprocessing import Process, Pipe
from argparse import ArgumentParser
import time
import sys, os
import traceback
from mlgame.wrapper import wrap_env


def get_screen(env):
    t = env.render()
    t = np.transpose(t, (2, 0, 1))
    t = torch.from_numpy(t)
    t = torch.unsqueeze(t, dim=0)
    return t

def np2tensor(arr):

    t = torch.from_numpy(arr)
    t = torch.unsqueeze(t, dim=0)
    # print(t.shape)
    return t


def env_entry_point(game_cls, recv, send, sz=10, num_frames=3):
    executor = GameRLModeExecutor(game_cls, recv, send)
    executor.start()



def wait_for_env(recv):
    while True:
        t = recv.recv()
        if t != None:
            return t 


def build_parser():


    parser = ArgumentParser()

    parser.add_argument("-o", "--output_dir", required=True, help="Required, where to save the model.", type=str)
    parser.add_argument("-m", "--model_src", help="[Optional] Pretrained model source", type=str)
    parser.add_argument("-i", "--iteration", required=True, help="The number of epoch to train the model", type=int)
    parser.add_argument("-b", "--batch_size", required=True, help="Batch size", type=int)
    parser.add_argument("-c", "--comment", required=False, help="The description of this training")
    return parser

def time2str(time :time.struct_time):

    return f"{time.tm_year}-{time.tm_mon}{time.tm_mday}-{time.tm_hour}-{time.tm_min}-{time.tm_sec}"





if __name__ == "__main__":

    parser = build_parser()
    args = parser.parse_args()



    num_episodes = args.iteration
    DEVICE = 'cuda'
    TARGET_UPDATE = 10
    cur_time = time.gmtime()

    recv_for_rl, send_for_env = Pipe(False)
    recv_for_env, send_for_rl = Pipe(False)

    # Start the process
    env_process = Process(target = env_entry_point, 
        name = 'Env_process', args = (Snake, recv_for_env, send_for_env))
    env_process.start()


    h, w = (10, 10)
    pretrained_src = args.model_src if args.model_src != None else None
    output_dir = f"{args.output_dir}"
    batch_size = args.batch_size

    dqnTrainer = dqn.DQNExecutor((h, w), 4, output_dir, device=DEVICE, pretrained_src=pretrained_src, batch_size=batch_size)

    try:
        loss = 0
        pbar = tqdm(range(num_episodes))


        ls = [0, 0, 0, 0, 0]
        for i_episode in pbar:
            i = 0
            frame_count = 0

            send_for_rl.send("READY")
            cur_state, reward, done, info = recv_for_rl.recv()
            cur_state = np2tensor(cur_state)
            next_state = None
            env_info = None
            loss = 0
            fitness = 0
            while True:
                i += 1
                action = dqnTrainer.select_action(cur_state)
                send_for_rl.send(action.item())
                # reward, done = env.step(action.item())
                frame_count += 1
                observation, reward, done, info = wait_for_env(recv_for_rl)
                next_state, reward, done, score = np2tensor(observation), \
                                           torch.tensor([reward], device=DEVICE), \
                                           done, \
                                           info['score']
                                            
                agent_frame = info['frame']
                
                dqnTrainer.push(cur_state, action, next_state, reward)
                cur_state = next_state
                fitness += reward.item()

                ls[action.item()] += 1
                loss = dqnTrainer.optimize_model()
                if done:
                    dqnTrainer.push_to_memory()
                    dqnTrainer.record(score, fitness, i_episode, eval=False)
                    if i_episode <= 2000:
                        dqnTrainer.steps_done = 0
                    pbar.set_postfix({'loss' : loss, 'actions': ls, 'sample_data' : f"{len(dqnTrainer.memory):e}"})
                    frame_count = 0
                    break
                
                delay_frame = agent_frame - frame_count
                if delay_frame > 0:
                    print(f"delay {delay_frame}")
                elif delay_frame == 0:
                    pass
                frame_count += 1
            
            if i_episode % TARGET_UPDATE == 0:
                dqnTrainer.update_network()

    except Exception as err:
        traceback.print_exception(*sys.exc_info())
    finally:
        dqnTrainer.save_model_directly()
        send_for_rl.send("QUIT")
        print("Model saved")

        

        
                
            



            



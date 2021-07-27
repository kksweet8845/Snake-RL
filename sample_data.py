from os import pardir
from mlgame.env import GameRLModeExecutor
from games.snake.game.snake import Snake
from r_learning import DQN_trainer as dqn
import sys, os
import logging as log
from argparse import ArgumentParser


def build_parser():
    parser = ArgumentParser()

    parser.add_argument("-o", "--output_dir", required=True, help="Required, where to save the model.", type=str)
    parser.add_argument("-m", "--model_src", help="[Optional] Pretrained model source", type=str)
    parser.add_argument("-i", "--iteration", required=True, help="The number of epoch to train the model", type=int)
    parser.add_argument("-b", "--batch_size", required=True, help="Batch size", type=int)
    parser.add_argument("-c", "--comment", required=False, help="The description of this training", type=str)
    # parser.add_argument("-g", "--gen_from_scratch", required=True, help="Generate network from scratch or not", type=int)
    return parser


def env_entry_point(game_cls, recv, send):

    env = GameRLModeExecutor(game_cls, recv, send)
    env.start()

if __name__ == "__main__":

    try:
        log.basicConfig(format="[%(levelname)s] %(message)s", level=log.INFO, stream=sys.stdout)
        parser = build_parser()
        args = parser.parse_args()

        input_shape = (300, 300)

        pretrained_src = args.model_src if args.model_src != None else None
        output_dir = f"{args.output_dir}"
        batch_size = args.batch_size

        os.makedirs(output_dir, exist_ok=True)

        DEVICE = 'cuda'

        trainer = dqn.RLTrainer(env_entry_point, Snake, input_shape, 4, output_dir,
                                batch_size=batch_size, pretrained_src=pretrained_src, device=DEVICE)

        # print(args.gen_from_scratch)
        arr = None
        for i in range(4):
            log.info("Generating n sample models")
            trainer.sample_data(args.iteration)

    finally:
        # trainer.dump_memory()
        # print("Memory dumped")
        trainer.close_env()
        print("Environment closed")



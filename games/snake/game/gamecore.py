"""
The file for managing gameobjects in the scene
"""

import random
import math
from pygame import Rect
from pygame.sprite import Group

from mlgame.utils.enum import StringEnum, auto

from .gameobject import Snake, Food
import numpy as np

class GameStatus(StringEnum):
    GAME_OVER = auto()
    GAME_ALIVE = auto()
    GAME_ATE_APPLE = auto()

class Scene:
    """
    The main game scene
    """

    high = 100
    width = 100
    # area_rect = Rect(0, 0, 300, 300)
    area_rect = Rect(0, 0, 100, 100)

    def __init__(self):
        self._create_scene()

        self.score = 0
        self._frame = 0
        self.height = 100
        self.width = 100
        self._status = GameStatus.GAME_ALIVE

    def _create_scene(self):
        """
        Import gameobjects to the scene and add them to the draw group
        """
        self._snake = Snake()
        self._food = [Food() for i in range(1)]
        # self._food = Food()
        self._random_all_food_pos()

        self._draw_group = Group()
        self._draw_group.add(self._snake.head, *self._snake.body, self._food)

    def _random_all_food_pos(self):
        """
        Randomly set the position of the food
        """

        for f in self._food:
            while True:
                candidate_pos = (
                    random.randrange(0, Scene.area_rect.width, 10),
                    random.randrange(0, Scene.area_rect.height, 10))

                if (candidate_pos != self._snake.head_pos and
                    not self._snake.is_body_pos(candidate_pos)):
                    break
            f.pos = candidate_pos

        # self._food.pos = candidate_pos

    def _random_food_pos(self, f):

        while True:
            candidate_pos = (
                random.randrange(0, Scene.area_rect.width, 10),
                random.randrange(0, Scene.area_rect.height, 10))

            if (candidate_pos != self._snake.head_pos and
                not self._snake.is_body_pos(candidate_pos)):
                break
        
        f.pos = candidate_pos


    def reset(self):
        self.score = 0
        self._frame = 0
        self._status = GameStatus.GAME_ALIVE

        self._snake = Snake()
        self._random_all_food_pos()
        self._draw_group.empty()
        self._draw_group.add(self._snake.head, *self._snake.body, self._food)

    def draw_gameobjects(self, surface):
        """
        Draw gameobjects to the given surface
        """
        self._draw_group.draw(surface)

    def update(self, action):
        """
        Update the scene

        @param action The action for controlling the movement of the snake
        """
        self._frame += 1
        self._snake.move(action)
        self._status = GameStatus.GAME_ALIVE

        for f in self._food:
            if self._snake.head_pos == f.pos:
                self.score += 1
                self._random_food_pos(f)
                new_body = self._snake.grow()
                self._draw_group.add(new_body)
                self._status = GameStatus.GAME_ATE_APPLE

        if (not Scene.area_rect.collidepoint(self._snake.head_pos) or
            self._snake.is_body_pos(self._snake.head_pos)):
            self._status = GameStatus.GAME_OVER

        return self._status, self.food_dis()

    def get_2d_pixel_from_scene(self):

        # scene = np.zeros((3, 300, 300), dtype=np.float32)
        height = 10
        width = 10
        scene = np.zeros((height, width, 3), dtype=np.float32)
        # set snake head
        n = np.clip(self._snake.head.pos, 0, height)
        n = n//10
        scene[n[0], n[1], 0] = 31
        scene[n[0], n[1], 1] = 204
        scene[n[0], n[1], 2] = 42

        # Set snake body
        for body in self._snake.body:
            n = np.clip(body.pos, 0, height)
            n = n//10
            scene[n[0], n[1], 0] = 255
            scene[n[0], n[1], 1] = 255
            scene[n[0], n[1], 2] = 255

        # Set snake food
        # left, top = self._food.pos
        for f in self._food:
            n = np.clip(f.pos, 0, height)
            n = n//10
            scene[n[0], n[1], 0] = 232
            scene[n[0], n[1], 1] = 54
            scene[n[0], n[1], 2] = 42

        return scene

    def food_dis(self):

        hx, hy = self._snake.head_pos
        fx, fy = self._food[0].pos

        l = (hx-fx)**2 + (hy - fy)**2

        return math.pow(l, 0.5)
        
        
    def get_scene_info(self):
        """
        Get the current scene information
        """
        scene_info = {
            "frame": self._frame,
            "status": self._status.value,
            "snake_head": self._snake.head_pos,
            "snake_body": [body.pos for body in self._snake.body],
            "snake_tail" : self._snake.body[-1].pos,
            "food": [ food.pos for food in self._food ],
            "action" : self._snake._action,
            "score" : self.score
        }

        return scene_info

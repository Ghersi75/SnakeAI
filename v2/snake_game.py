### Code from https://github.com/patrickloeber/python-fun/tree/master/snake-pygame

import pygame
import random
from enum import Enum
from collections import namedtuple
import os
import numpy as np

pygame.init()

font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 150

WIDTH = 640
HEIGHT = 480

class SnakeGameAI:
    def __init__(self, w=WIDTH, h=HEIGHT, n_snakes=1):
        self.w = w
        self.h = h
        # Add option to have multiple snakes, and initialize each one to be a list of None elements for each snake
        # These value should all be set once the reset() function is called
        self.n_snakes = n_snakes
        # All snakes start facing right
        self.directions = [Direction.RIGHT] * n_snakes
        self.heads = [None] * n_snakes
        self.snakes = [None] * n_snakes
        # All snakes start with 0 score
        self.scores = [0] * n_snakes
        # All snakes start with no food set
        self.foods = [None] * n_snakes
        # All snakes start with 0 iterations played
        self.frame_iterations = [0] * n_snakes
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        self.game_overs = [False] * n_snakes
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        # self.reset()
        
    # init game state
    # Should reset all values regardless of whether or not init values are the same
    def reset(self):
        for i in range(self.n_snakes):
            self.directions[i] = Direction.RIGHT
            
            self.heads[i] = Point(self.w/2, self.h/2)
            self.snakes[i] = [self.heads[i], 
                        Point(self.heads[i].x-BLOCK_SIZE, self.heads[i].y),
                        Point(self.heads[i].x-(2*BLOCK_SIZE), self.heads[i].y)]
            
            self.scores[i] = 0
            self.foods[i] = None
            self._place_food(i)
            self.frame_iterations[i] = 0
        
    def _place_food(self, i):
        x = random.randint(0, (self.w-BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.foods[i] = Point(x, y)
        if self.foods[i] in self.snakes[i]:
            self._place_food(i)
        
    # This function simply makes the next step with action as the direction it should be going in, and it returns the reward, whether the game is over, and the score
    def play_step(self, action, i):
        self.frame_iterations[i] += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action, i) # update the head
        self.snakes[i].insert(0, self.heads[i])
        
        # 3. check if game over
        reward = 0
        # If snake collides or it doesnt do anything for too long, end game
        if self.is_collision(i) or self.frame_iterations[i] > 100 * len(self.snakes[i]):
            self.game_overs[i] = True
            reward = -10
            return reward, self.game_overs[i], self.scores[i]
            
        # 4. place new food or just move
        if self.heads[i] == self.foods[i]:
            self.scores[i] += 1
            reward += 10
            self._place_food(i)
        else:
            self.snakes[i].pop()
        
        # 5. update ui and clock
        # TODO make this work for each idx
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        return reward, self.game_overs[i], self.scores[i]
    
    def is_collision(self, i, point=None):
        # If there's no given point to check collision for, use the head
        if point == None:
            point = self.heads[i]
        # hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return True
        # hits itself
        if point in self.snakes[i][1:]:
            return True
        
        return False
    
    # TODO make this work for each idx
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for snake in self.snakes:
            for pt in snake:
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        for food in self.foods:
            pygame.draw.rect(self.display, RED, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Scores: " + str(self.scores), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action, i):
        # straight, right, left
        # print(action)
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)
        # 1, 0, 0 go straight, no direction change
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clockwise[idx]
        # 0, 1, 0 make a right turn, clockwise turn
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clockwise[next_idx]
        # 0, 0, 1 make a left turn, counter clockwise turn
        # np.array_equal(action, [0, 0, 1])
        else:
            next_idx = (idx - 1) % 4
            new_dir = clockwise[next_idx]
        
        self.directions[i] = new_dir
        x = self.heads[i].x
        y = self.heads[i].y
        if new_dir == Direction.RIGHT:
            x += BLOCK_SIZE
        elif new_dir == Direction.LEFT:
            x -= BLOCK_SIZE
        elif new_dir == Direction.DOWN:
            y += BLOCK_SIZE
        elif new_dir == Direction.UP:
            y -= BLOCK_SIZE
            
        self.heads[i] = Point(x, y)

    def test(self):
        self.reset()
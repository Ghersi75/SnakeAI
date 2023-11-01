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

class Snake:
    def __init__(self, direction, head, snake, score, food, gameOver, model):
        self.direction = direction
        self.head = head
        self.snake = snake
        self.score = score
        self.food = food
        self.gameOver = gameOver
        self.model = model
        self.frameIterations = 0
    
    def getDirection(self):
        return self.direction

    def setDirection(self, newDir):
        self.direction = newDir

    def getHead(self):
        return self.head

    def setHead(self, newHead):
        self.head = newHead

    def getSnake(self):
        return self.snake

    def setSnake(self, newSnake):
        self.snake = newSnake

    def getScore(self):
        return self.score

    def setScore(self, newScore):
        self.setScore = newScore

    def getFood(self):
        return self.food

    def setFood(self, newFood):
        self.setFood = newFood
    
    def getGameOver(self):
        return self.gameOver

    def setGameOver(self, newGameOver):
        self.game_over = newGameOver

    def getModel(self):
        return self.model

    def setModel(self, newModel):
        self.model = newModel

    def getFrameIterations(self):
        return self.frameIterations

    def setFrameIterations(self, newIterations):
        self.frameIterations = newIterations

class SnakeGameAI:
    def __init__(self, w=WIDTH, h=HEIGHT, n_snakes=1):
        self.w = w
        self.h = h
        self.n_snakes = n_snakes
        self.snakes = []
        for i in range(n_snakes):
            new_snake = Snake(Direction.RIGHT,  # Direction
                              None,             # Head
                              None,             # Full snake, body and head
                              0,                # Score
                              None,             # Food
                              False,            # Game over, false by default
                              None)             # Neural network model, none by default
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        # self.reset()
        
    # init game state
    def reset(self, models):
        if len(models) != self.n_snakes:
            # Just in case I forget
            raise Exception("Number of models given does not match number of snakes")
        
        for i in range(self.n_snakes):
            # Not sure if doing something like snake = self.snakes[i] would be reference, but it should be since its a class reference
            currSnake = self.snakes[i]
            currSnake.setDirection(Direction.RIGHT)
            
            currSnake.setHead(Point(self.w/2, self.h/2))
            # This should also be fine
            currHead = currSnake.getHead()
            currSnake.setSnake([currHead, 
                                Point(currHead.x-BLOCK_SIZE, currHead.y),
                                Point(currHead.x-(2*BLOCK_SIZE), currHead.y)])
            currSnake.setScore(0)
            currSnake.setFood(None)
            self._place_food(i)
            currSnake.setModel(models[i])
            currSnake.setGameOver(False)
            currSnake.setFrameIterations(0)
        
    def _place_food(self, i):
        currSnake = self.snakes[i]
        x = random.randint(0, (self.w-BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        food = Point(x, y)
        if food in currSnake.getSnake():
            self._place_food(i)
        else:
            currSnake.setFood(food)

    # TODO fix
    # This function simply makes the next step with action as the direction it should be going in, and it returns the reward, whether the game is over, and the score
    def play_step(self, action, i):
        currSnake = self.snakes[i]
        currSnake.setFrameIterations(currSnake.getFrameIterations() + 1)
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
    
    # TODO fix
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
        
    # TODO fix
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
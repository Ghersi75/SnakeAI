### Code from https://github.com/patrickloeber/python-fun/tree/master/snake-pygame

import pygame
import random
from enum import Enum
from collections import namedtuple
from config import *
from Snake import Snake
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
BLUE = (0, 0, 200)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN = (0, 200, 0)
GREEN1 = (0, 255, 0)
GREEN2 = (100, 255, 0)
BLACK = (0,0,0)

class SnakeGameAI:
    """
    game should be in the following shape:
    type game = {
        height: number,
        width: number,
        gameMoves: {        
                snakeBody: Point[]
                food: Point
            }[]
    }
    These are the basics variables needed 
    """    
    def __init__(self, game, s=SPEED):
        if len(game['gameMoves']) == 0:
            raise Exception("No game moves given")
        try:
            self.w = game['width']
            self.h = game['height']
            self.currMove = 0
            firstMove = game['gameMoves'][0]
            self.snake = firstMove['snakeBody']
            self.food = firstMove['food']
                                
            self.display = pygame.display.set_mode((self.w, self.h), pygame.NOFRAME)
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()
        except:
            raise Exception("Missing game data")
        
    def updateUi(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
            food = self.food
            pygame.draw.rect(self.display, BLUE, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.display.flip()
        self.clock.tick(self.s)
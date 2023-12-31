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
    def __init__(self, numSnakes, w=WIDTH, h=HEIGHT):
        self.w = w
        self.h = h
        self.numSnakes = numSnakes
        self.snakes = []
        for i in range(numSnakes):
            newSnake = Snake(Direction.RIGHT,  # Direction
                              None,             # Head
                              None,             # Full snake, body and head
                              0,                # Score
                              None,             # Food
                              False,            # Game over, false by default
                              None)             # Neural network model, none by default
            self.snakes.append(newSnake)
        # init display
        self.display = pygame.display.set_mode((self.w, self.h), pygame.NOFRAME)
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        # self.reset()

    def getSnake(self, i):
        return self.snakes[i]

    # init game state
    def reset(self, models):
        if len(models) != self.numSnakes:
            # Just in case I forget
            raise Exception("Number of models given does not match number of snakes")
        
        for i in range(self.numSnakes):
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
            self._placeFood(i)
            currSnake.setModel(models[i])
            currSnake.setGameOver(False)
            currSnake.setFrameIterations(0)
            # Just in case
            self.snakes[i] = currSnake
        
    def _placeFood(self, i):
        currSnake = self.snakes[i]
        x = random.randint(0, (self.w-BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        food = Point(x, y)
        if food in currSnake.getSnake():
            self._placeFood(i)
        else:
            currSnake.setFood(food)

    def playStep(self, action, i):
        currSnake = self.snakes[i]
        currSnake.setFrameIterations(currSnake.getFrameIterations() + 1)
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action, i) # update the head
        # Insert the head at index 0
        # Removing the last element if no food was picked up is handled next
        currSnake.setSnake([currSnake.getHead()] + currSnake.getSnake())

        # 3. check if game over
        # If snake collides or it doesnt do anything for too long, end game
        if currSnake.getSnake() != None and self.isCollision(i) or currSnake.getFrameIterations() > AMOUNT_OF_FRAMES_TO_DEATH_MULTIPLIER * len(currSnake.getSnake()):
            currSnake.setGameOver(True)
            currSnake.setFinalLength(len(currSnake.getSnake()))
            currSnake.setSnake([])
            currSnake.setHead(None)
            if currSnake.getFrameIterations() > AMOUNT_OF_FRAMES_TO_DEATH_MULTIPLIER * currSnake.getFinalLength():
                # Lazy death
                currSnake.setDeath(0)

        # 4. place new food or just move
        if currSnake.getHead() == currSnake.getFood():
            currSnake.setScore(currSnake.getScore() + 1)
            self._placeFood(i)
        else:
            newSnake = currSnake.getSnake()
            if newSnake is None or len(newSnake) == 0:
                return
            newSnake.pop()
            currSnake.setSnake(newSnake)
    
    def isCollision(self, i, point=None):
        currSnake = self.snakes[i]
        # If there's no given point to check collision for, use the head
        if point == None:
            point = currSnake.getHead()
        # hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            # Hit the wall
            currSnake.setDeath(1)
            return True
        # hits itself
        elif currSnake.getSnake() is not None and len(currSnake.getSnake()) > 0 and point in currSnake.getSnake()[1:]:
            # Hit itself
            currSnake.setDeath(2)
            return True
        
        return False
    
    # This will be called by agent after all snakes made their move
    def updateUi(self):
        self.display.fill(BLACK)
        # Only display the first model, aka the child of the previous 2 best
        firstSnakeAlive = self.snakes[0]
        # idx = None
        for idx in range(len(self.snakes)):
            if not self.snakes[idx].getGameOver():
                firstSnakeAlive = self.snakes[idx]
                break
        # Same index as first one alive from above
        bestSnakeAlive = self.snakes[idx]
        for bestIdx in range(len(self.snakes)):
            currSnake = self.snakes[bestIdx]
            if currSnake.getScore() > bestSnakeAlive.getScore():
                if not currSnake.getGameOver():
                    bestSnakeAlive = currSnake
        for pt in firstSnakeAlive.getSnake():
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
            food = firstSnakeAlive.getFood()
            pygame.draw.rect(self.display, BLUE, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        for pt in bestSnakeAlive.getSnake():
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
            food = bestSnakeAlive.getFood()
            pygame.draw.rect(self.display, GREEN, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))
        

        scores = []
        for i in range(self.numSnakes):
            scores.append(self.snakes[i].getScore())
        # FirstSnakeAliveText = font.render(f"Snake {idx}'s Score: {firstSnakeAlive.getScore()}", True, WHITE)
        # CurrBestAliveText = font.render(f"Best Alive Score ({self.snakes.index(bestSnakeAlive)}): {bestSnakeAlive.getScore()}", True, WHITE)
        # CurrentGenerationMaxScore = font.render(f"Current Generation Max Score: {max(scores)}", True, WHITE)

        # #  - 
        # self.display.blit(FirstSnakeAliveText, [0, 0 * BLOCK_SIZE])
        # self.display.blit(CurrBestAliveText, [0, 1 * BLOCK_SIZE])
        # self.display.blit(CurrentGenerationMaxScore, [0, 2 * BLOCK_SIZE])
        pygame.display.flip()
        self.clock.tick(SPEED)

    def _move(self, action, i):
        currSnake = self.snakes[i]
        # straight, right, left
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(currSnake.getDirection())
        # 1, 0, 0 go straight, no direction change
        if np.array_equal(action, [1, 0, 0]):
            newDir = clockwise[idx]
        # 0, 1, 0 make a right turn, clockwise turn
        elif np.array_equal(action, [0, 1, 0]):
            nextIdx = (idx + 1) % 4
            newDir = clockwise[nextIdx]
        # 0, 0, 1 make a left turn, counter clockwise turn
        # np.array_equal(action, [0, 0, 1])
        else:
            nextIdx = (idx - 1) % 4
            newDir = clockwise[nextIdx]
        
        currSnake.setDirection(newDir)
        currHead = currSnake.getHead()
        x = currHead.x
        y = currHead.y
        if newDir == Direction.RIGHT:
            x += BLOCK_SIZE
        elif newDir == Direction.LEFT:
            x -= BLOCK_SIZE
        elif newDir == Direction.DOWN:
            y += BLOCK_SIZE
        elif newDir == Direction.UP:
            y -= BLOCK_SIZE
            
        currSnake.setHead(Point(x, y))

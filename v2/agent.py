import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot, model_folder_path, SNAKE_VISION_RADIUS
import os

# Agent class used to manage the game and AI
class Agent:
    # TODO look at this one
    def __init__(self, numSnakes=1):
        self.nGames = 0
        self.numSnakes = numSnakes
        self.game = SnakeGameAI()

    def getState(self, i):
        game = self.game
        currSnake = game.getSnake(i)
        head = currSnake.getHead()
        # Get 9x9 grid around snake for model inputs
        snakeVisionArr = []
        for i in range(-1 * SNAKE_VISION_RADIUS, SNAKE_VISION_RADIUS + 1):
            for j in range(-1 * SNAKE_VISION_RADIUS, SNAKE_VISION_RADIUS + 1):
                # If either of these is negative, is_collision will return true
                checkX = head.x + BLOCK_SIZE * i
                checkY = head.x + BLOCK_SIZE * j
                checkPoint = Point(checkX, checkY)
                if game.isCollision(i, checkPoint):
                    # If the given point would cause a collision, meaning the wall or snake, append -1
                    snakeVisionArr.append(-1)
                elif game.foods[i] == checkPoint:
                    # If the point given is where the food is append 1
                    snakeVisionArr.append(1)
                else:
                    # If its not going to cause a collision or be food, append 0
                    snakeVisionArr.append(0)

        food = currSnake.getFood()
        direction = currSnake.getDirection()
        # Get 4 points around the head
        pointLeft = Point(head.x - BLOCK_SIZE, head.y)
        pointRight = Point(head.x + BLOCK_SIZE, head.y)
        pointUp = Point(head.x, head.y - BLOCK_SIZE)
        pointDown = Point(head.x, head.y + BLOCK_SIZE)

        # Get which direction the head is going towards
        dirLeft = direction == Direction.LEFT
        dirRight = direction == Direction.RIGHT
        dirUp = direction == Direction.UP
        dirDown = direction == Direction.DOWN

        state = snakeVisionArr + [
            # Danger straight ahead
            (dirRight and game.isCollision(i, pointRight)) or 
            (dirLeft and game.isCollision(i, pointLeft)) or 
            (dirUp and game.isCollision(i, pointUp)) or 
            (dirDown and game.isCollision(i, pointDown)),

            # Danger to the right
            (dirRight and game.isCollision(i, pointDown)) or 
            (dirLeft and game.isCollision(i, pointUp)) or 
            (dirUp and game.isCollision(i, pointRight)) or 
            (dirDown and game.isCollision(i, pointLeft)),

            # Danger to the left
            (dirRight and game.isCollision(i, pointUp)) or 
            (dirLeft and game.isCollision(i, pointDown)) or 
            (dirUp and game.isCollision(i, pointLeft)) or 
            (dirDown and game.isCollision(i, pointRight)),

            # Move directions
            dirLeft,
            dirRight,
            dirUp,
            dirDown,

            # Food direction
            food.x < head.x, # Food is left of us
            food.x > head.x, # Food is right of us
            food.y < head.y, # Food is above us - y 0 is at the top y max is at the bottom
            food.y > head.y  # Food is below us
        ]

        return np.array(state, dtype=int)

    # No need for guessing or idx input
    # Evolution essentially just guesses over and over until it gets good at it, so no need to hardcode guessing
    def getAction(self, i, state):
        nextMove = [0, 0, 0]
        stateTensor = torch.tensor(state, dtype=torch.float)
        model = self.game.getSnake(i).getModel()
        prediction = model(stateTensor)
        # argmax will return the index of the highest number
        # [0, 1, 0] will return 1
        # [1, 0, 0] will return 0
        # [99, 98, 100] will return 2
        move = torch.argmax(prediction).item()
        nextMove[move] = 1

        return nextMove
    
    def train(self):
        while True:
            gameOvers = [False] * self.numSnakes
            for i in range(self.numSnakes):
                currSnake = self.game.getSnake(i)
                # If current snake's game hasn't ended, get a move and keep playing
                if not currSnake.getGameOver():
                    currState = self.getState(i)
                    nextAction = self.getAction(i, currState)
                    self.game.playStep(nextAction, i)
                else:
                    gameOvers[i] = True
            
            if gameOvers.count(True) == self.numSnakes:
                break
        
        # TODO Logic for generation evolution
        

if __name__ == "__main__":
    agent = Agent()
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
    def __init__(self, nSnakes=1):
        self.nGames = 0
        self.nSnakes = nSnakes
        self.game = SnakeGameAI()

    def getState(self, i):
        game = self.game
        head = game.getHead(i)
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

        food = game.foods[i]
        direction = game.directions[i]
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
    def getAction(self, state):
        nextMove = [0, 0, 0]

        state0 = torch.tensor(state, dtype=torch.float)
        # This will automatically call the forward function
        prediction = self.model(state0)
        # argmax will return the index of the highest number
        # [0, 1, 0] will return 1
        # [1, 0, 0] will return 0
        # [99, 98, 100] will return 2
        move = torch.argmax(prediction).item()
        nextMove[move] = 1

        return nextMove

    # TODO look at this one
def train(agent):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    game = SnakeGameAI()
    while True:
        for i in range(agent.n_games):
            # Get old/curr state
            state_old = agent.get_state(game, i)

            # Get next move
            next_move = agent.get_action(state_old)

            # Perform move and get new state
            reward, game_over, score = game.play_step(next_move, i)
            state_new = agent.get_state(game, i)

            # Train short memory
            agent.train_short_memory(state_old, next_move, reward, state_new, game_over)

            # Store current state in deque
            agent.remember(state_old, next_move, reward, state_new, game_over)

            if game_over:
                # Train long memory / Replay training and plot the results
                # game.reset()
                # agent.n_games += 1
                # agent.train_long_memory()

                # If score > record, record = score
                if score > record:
                    record = score
                    agent.model.save(agent.n_games)

        print(f"Game: {agent.n_games}, Score: {score}, Record: {record}")
        # TODO fix these values to take only best result and set after all games are over
        plot_scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games
        plot_mean_scores.append(mean_score)
        plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    agent = Agent()

    train(agent)
import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot, model_folder_path, SNAKE_VISION_RADIUS
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
# Learning rate
LR = 1e-3

# Agent class used to manage the game and AI
class Agent:
    # TODO look at this one
    def __init__(self, n_snakes=1):
        self.n_games = 0
        self.n_snakes = n_snakes
        if os.path.exists(os.path.join(model_folder_path, "model.pth")):
            model_state = torch.load(os.path.join(model_folder_path, "model.pth"))
            self.n_games = model_state['n_games']
        # Randomness
        self.epsilon = 0 
        # Discount rate used for the magic Q function
        self.gamma = .9 # Must be smaller than 1
        # Removes elements from the left if it fills up
        self.memory = deque(maxlen=MAX_MEMORY)
        # 11 inputs
        # 256 hidden neurons
        # 3 Outputs, straight, left, right
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) 

    def get_state(self, game, i):
        snake_vision_arr = []
        for i in range(-1 * SNAKE_VISION_RADIUS, SNAKE_VISION_RADIUS + 1):
            for j in range(-1 * SNAKE_VISION_RADIUS, SNAKE_VISION_RADIUS + 1):
                # If either of these is negative, is_collision will return true
                check_x = game.heads[i].x + BLOCK_SIZE * i
                check_y = game.heads[i].x + BLOCK_SIZE * j
                check_point = Point(check_x, check_y)
                if game.is_collision(i, check_point):
                    # If the given point would cause a collision, meaning the wall or snake, append -1
                    snake_vision_arr.append(-1)
                elif game.foods[i] == check_point:
                    # If the point given is where the food is append 1
                    snake_vision_arr.append(1)
                else:
                    # If its not going to cause a collision or be food, append 0
                    snake_vision_arr.append(0)

        head = game.heads[i]
        food = game.foods[i]
        direction = game.directions[i]
        # Get 4 points around the head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        # Get which direction the head is going towards
        dir_l = direction == Direction.LEFT
        dir_r = direction == Direction.RIGHT
        dir_u = direction == Direction.UP
        dir_d = direction == Direction.DOWN

        state = snake_vision_arr + [
            # Danger straight ahead
            (dir_r and game.is_collision(i, point_r)) or 
            (dir_l and game.is_collision(i, point_l)) or 
            (dir_u and game.is_collision(i, point_u)) or 
            (dir_d and game.is_collision(i, point_d)),

            # Danger to the right
            (dir_r and game.is_collision(i, point_d)) or 
            (dir_l and game.is_collision(i, point_u)) or 
            (dir_u and game.is_collision(i, point_r)) or 
            (dir_d and game.is_collision(i, point_l)),

            # Danger to the left
            (dir_r and game.is_collision(i, point_u)) or 
            (dir_l and game.is_collision(i, point_d)) or 
            (dir_u and game.is_collision(i, point_l)) or 
            (dir_d and game.is_collision(i, point_r)),

            # Move directions
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food direction
            food.x < head.x, # Food is left of us
            food.x > head.x, # Food is right of us
            food.y < head.y, # Food is above us - y 0 is at the top y max is at the bottom
            food.y > head.y  # Food is below us
        ]

        return np.array(state, dtype=int)

    # TODO look at this one - should be good
    # Store current state, not sure why it was named remember instead of something better
    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    # TODO look at this one
    # This will train multiple games over time
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            # This will return BATCH_SIZE amount of tuples 
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)


    # TODO look at this one
    # This function will train 1 step at a time
    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    # TODO look at this one
    def get_action(self, state, i):
        # random moves: tradeoff exploration vs exploitation
        # essentially, start as random, and move to predicted moves as the model improves
        self.epsilon = 50 - self.n_games
        next_move = [0, 0, 0]
        # As more games happen, epsilon decreases, and less random moves are done
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            next_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            # This will automatically call the forward function
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            next_move[move] = 1

        return next_move

    # TODO look at this one
def train(agent):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    game = SnakeGameAI()
    while True:
        # Get old/curr state
        state_old = agent.get_state(game)

        # Get next move
        next_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, game_over, score = game.play_step(next_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, next_move, reward, state_new, game_over)

        # Remember
        agent.remember(state_old, next_move, reward, state_new, game_over)

        if game_over:
            # Train long memory / Replay training and plot the results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # If score > record, record = score
            if score > record:
                record = score
                agent.model.save(agent.n_games)

            print(f"Game: {agent.n_games}, Score: {score}, Record: {record}")

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    threads = 5
    agent = Agent()
    # ShareResources()

    train(agent)
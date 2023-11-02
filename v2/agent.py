import torch
import numpy as np
from snake_game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import EvolutionNetwork, averageCrossover, mutateModel
from helper import SNAKE_VISION_RADIUS, AMOUNT_OF_FRAMES_TO_DEATH_MULTIPLIER

# Agent class used to manage the game and AI
class Agent:
    # TODO look at this one
    def __init__(self, numSnakes):
        self.nGames = 0
        self.numSnakes = numSnakes
        self.game = SnakeGameAI(numSnakes)
        # Each model should have randomize weights and biases, so each model should be different at the beginning
        models = [EvolutionNetwork(92, 256, 3) for i in range(numSnakes)]
        self.game.reset(models=models)

    def getState(self, i):
        game = self.game
        currSnake = game.getSnake(i)
        head = currSnake.getHead()
        # Get 9x9 grid around snake for model inputs
        snakeVisionArr = []
        for j in range(-1 * SNAKE_VISION_RADIUS, SNAKE_VISION_RADIUS + 1):
            for k in range(-1 * SNAKE_VISION_RADIUS, SNAKE_VISION_RADIUS + 1):
                # If either of these is negative, is_collision will return true
                checkX = head.x + BLOCK_SIZE * j
                checkY = head.y + BLOCK_SIZE * k
                checkPoint = Point(checkX, checkY)
                if game.isCollision(i, checkPoint):
                    # If the given point would cause a collision, meaning the wall or snake, append -1
                    snakeVisionArr.append(-1)
                elif currSnake.getFood() == checkPoint:
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
    
    # Finds score of given model index and returns
    def fitnessFunction(self, i):
        currSnake = self.game.getSnake(i)
        score = currSnake.getScore() 
        frameIterations = currSnake.getFrameIterations()
        maxIterations = currSnake.getFinalLength() * AMOUNT_OF_FRAMES_TO_DEATH_MULTIPLIER + 1
        death = currSnake.getDeath()
        # Score will be based on score, how long it lasted, and how it died
        # The goal is the get the highest score obviously, but also avoid running into itself later down the line, which was an issue with Q Learning approach
        fitness = 0
        fitness += score * 10 # Biggest factor
        fitness += (frameIterations / maxIterations) * 50 # Shouldn't be a huge amount, but will matter at the beginning of the game
        if death == 0:
            # We don't like lazy
            # Cancels out fitness from surviving by doing nothing
            fitness -= 50
        elif death == 1:
            # We also don't like running into walls, but it's not a huge deal
            fitness -= 10
        elif death == 2:
            # We don't want it running into itself, but later in the game it will be harder
            fitness -= 50
        
        return fitness

    def train(self, generations=1):
        for gen in range(generations):
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

                self.game.updateUi()
                if gameOvers.count(True) == self.numSnakes:
                    # print(currSnake.getModel().state_dict())
                    break
        
            # TODO Logic for generation evolution
            fitness = []
            for i in range(self.numSnakes):
                currFitness = self.fitnessFunction(i)
                fitness.append(currFitness)
            sortedFitness = fitness[:]
            sortedFitness.sort(reverse=True)
            # print(fitness)
            # print(sortedFitness)
            # print(fitness.index(max(fitness)))
            parentAIndex = fitness.index(sortedFitness[0])
            parentBIndex = fitness.index(sortedFitness[0])

            parentA = self.game.getSnake(parentAIndex).getModel()
            parentB = self.game.getSnake(parentBIndex).getModel()

            child = averageCrossover(parentA, parentB)
            models = [child]
            
            for i in range(1, self.numSnakes):
                # We want each model to mutate
                model = mutateModel(child, mutationRate=1)
                models.append(model)
            
            print(f"Generation {gen + 1} done. Best fitness: {sortedFitness[0]}")
            self.game.reset(models)

if __name__ == "__main__":
    agent = Agent(100)

    agent.train(50)
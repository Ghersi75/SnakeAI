import torch
import numpy as np
from SnakeGames.SnakeGame import SnakeGameAI, Direction, Point, BLOCK_SIZE
from SnakeGames.SnakeGameNoGUI import SnakeGameNoGUI
from model import EvolutionNetwork, averageCrossover, mutateModel
from config import *
import time
from threading import Thread
import os

# Agent class used to manage the game and AI
class Agent:
    # TODO look at this one
    def __init__(self, numSnakes, modelLoadName=None):
        self.numSnakes = numSnakes
        if SHOW_GAME:
            self.game = SnakeGameAI(numSnakes)
        else:
            self.game = SnakeGameNoGUI(numSnakes)
        
        if modelLoadName and self.loadModel(modelLoadName):
            savedData = self.loadModel(modelLoadName)
            self.numGenerations = savedData['numGenerations']
            self.bestFitnessCurrGeneration = savedData['bestFitnessCurrGeneration']
            self.bestFitnessEver = savedData['bestFitnessEver']
            savedModelStateDicts = savedData['modelsStateDicts']
            print(f"Loaded Model\nNum Generations: {self.numGenerations}\nBest Fitness Ever: {self.bestFitnessEver}\nBest Fitness Last Generation: {self.bestFitnessCurrGeneration}")
            models = []
            for modelIdx in range(len(savedModelStateDicts)):
                currModel = EvolutionNetwork(92, 256, 3)
                currModel.load_state_dict(savedModelStateDicts[modelIdx])
                models.append(currModel)
        else:
            self.numGenerations = 0
            self.bestFitnessCurrGeneration = 0
            self.bestFitnessEver = 0
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
        fitness += score * 100 # Biggest factor
        fitness += (frameIterations / maxIterations) * 50 # Shouldn't be a huge amount, but will matter at the beginning of the game
        if death == 0:
            # We don't like lazy
            # Cancels out fitness from surviving by doing nothing
            fitness *= .9
        elif death == 1:
            # We also don't like running into walls, but it's not a huge deal
            fitness *= .9
        elif death == 2:
            # We don't want it running into itself, but later in the game it will be harder
            fitness *= .9
        
        return fitness

    # Multithreaded usage
    def trainBatch(self, gameOvers, gameSteps, batchRange):
        for idx in batchRange:
            # print(f"Batch: {batchRange}, idx: {idx}")
            self.trainIndividual(gameOvers, gameSteps, idx)

    def trainIndividual(self, gameOvers, gameSteps, idx):
        currSnake = self.game.getSnake(idx)
        # If current snake's game hasn't ended, get a move and keep playing
        if not currSnake.getGameOver():
            currState = self.getState(idx)
            nextAction = self.getAction(idx, currState)
            gameSteps[idx] = nextAction
        else:
            gameOvers[idx] = True

    def loadModel(self, modelLoadName):
        filePath = __file__
        currentWorkingDirectory = os.path.dirname(os.path.abspath(filePath))
        loadPath = os.path.join(currentWorkingDirectory, f"model\\{modelLoadName}.pth")
        if not os.path.exists(loadPath):
            return None
        
        savedData = torch.load(loadPath)
        return savedData

    def saveModel(self, modelSaveName=None):
        filePath = __file__
        currentWorkingDirectory = os.path.dirname(os.path.abspath(filePath))
        if modelSaveName == None:
            modelSaveName = f"{self.numSnakes}Model-{self.numGenerations + 1}-{self.bestFitnessEver:.2f}-{self.bestFitnessCurrGeneration:.2f}"
        savePath = os.path.join(currentWorkingDirectory, f"model\\{modelSaveName}.pth")
        modelsStateDicts = []
        for i in range(self.numSnakes):
            currModelStateDict = self.game.getSnake(i).getModel().state_dict()
            modelsStateDicts.append(currModelStateDict)
        saveDict = {
            'modelsStateDicts': modelsStateDicts,
            'numGenerations': self.numGenerations,
            'bestFitnessCurrGeneration': self.bestFitnessCurrGeneration,
            'bestFitnessEver': self.bestFitnessEver
        }
        torch.save(saveDict, savePath)

    def train(self, generations=1):
        for gen in range(generations):
            start = time.time()
            while True:
                gameOvers = [False] * self.numSnakes
                gameSteps = [None] * self.numSnakes
                numThreads = 10
                batchSize = self.numSnakes // numThreads 
                batchRanges = []
                # Start at 0, go to numSnakes, step batchSize at a time
                for i in range(numThreads):
                    if i == numThreads - 1:
                        # Last batch will take on extras since I somehow am struggling to split them up evenly
                        batchRange = range(i * batchSize, self.numSnakes)
                    else:
                        batchRange = range(i * batchSize, i * batchSize + batchSize)
                    batchRanges.append(batchRange)
                
                # print(batchRanges)
                # return
                threads = []
                for i in range(numThreads):
                    thread = Thread(target=self.trainBatch, args=(gameOvers, gameSteps, batchRanges[i]))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                for snakeIdx in range(self.numSnakes):
                    if gameSteps[snakeIdx] is not None:
                        self.game.playStep(gameSteps[snakeIdx], snakeIdx)

                # No need if theres no GUI
                if SHOW_GAME:
                    self.game.updateUi()
                if gameOvers.count(True) == self.numSnakes:
                    # print(currSnake.getModel().state_dict())
                    break
        
            # TODO Logic for generation evolution
            if self.numSnakes > 1:
                fitness = []
                for i in range(self.numSnakes):
                    currFitness = self.fitnessFunction(i)
                    fitness.append(currFitness)
                sortedFitness = fitness[:]
                sortedFitness.sort(reverse=True)
                self.bestFitnessCurrGeneration = sortedFitness[0]
                if sortedFitness[0] > self.bestFitnessEver:
                    self.bestFitnessEver = sortedFitness[0]
                # print(fitness)
                # print(sortedFitness)
                # print(fitness.index(max(fitness)))
                parentAIndex = fitness.index(sortedFitness[0])
                parentBIndex = fitness.index(sortedFitness[1])

                parentA = self.game.getSnake(parentAIndex).getModel()
                parentB = self.game.getSnake(parentBIndex).getModel()

                # Don't mutate first child
                child = averageCrossover(parentA, parentB, mutationRate=0)
                models = [child]
                # Get the best 10% from previous generation
                for currBestIdx in range(self.numSnakes // 10):
                    currBestId = fitness.index(sortedFitness[currBestIdx])
                    currBestModel = self.game.getSnake(currBestId).getModel()
                    models.append(currBestModel)
                l = len(models)
                # print(l) # numSnakes - 1 - numSnakes // 10
                for i in range(l, self.numSnakes):
                    # We want each model to mutate
                    model = mutateModel(child, mutationRate=1)
                    models.append(model)
            else:
                newModel = mutateModel(self.game.getSnake(0).getModel(), mutationRate=1)
                models = [newModel]
            # print(len(models)) # numSnakes
            end = time.time()
            print(f"Generation {self.numGenerations + gen + 1} done\n\tBest fitness: {sortedFitness[0]:.2f}\n\tMean fitness: {sum(sortedFitness) / len(sortedFitness):.2f}\n\tMedian fitness: {sortedFitness[len(sortedFitness) // 2]:.2f}\n\tWorst fitness: {sortedFitness[-1]:.2f}\n\tChild of previous gen's fitness: {fitness[0]:.2f}\n\tTime taken: {end - start:.2f}s")
            self.game.reset(models)

        self.numGenerations += generations
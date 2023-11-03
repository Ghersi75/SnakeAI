from agent import Agent
import sys

if __name__ == "__main__":
    args = iter(sys.argv[1:])
    numSnakes = int(next(args, 250))
    numGenerations = int(next(args, 100))
    modelLoadName = next(args, None)
    modelSaveName = next(args, None)

    agent = Agent(numSnakes, modelLoadName=modelLoadName)
    agent.train(numGenerations)
    agent.saveModel(modelSaveName)
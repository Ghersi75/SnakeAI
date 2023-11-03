import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random

class EvolutionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def mutateModel(originalModel, mutationRate=0.1, mutationStrength=0.1):
    # Create a deep copy of the original model
    mutated_model = copy.deepcopy(originalModel)

    # This iterates over all weights and biases of the network
    for param in mutated_model.parameters():
        # Mutate based on mutationRate
        # random.random generates a float between 0 and 1 inclusive
        if random.random() < mutationRate:
            # Add some noise, aka some random numbers to the whole network
            # randn_like creates a martix in the same shape as param with each index having a number between 0 and 1
            # Then multiple each one by the mutations strength to add small amounts of noise at a time
            noise = torch.randn_like(param) * mutationStrength
            param.data += noise
    
    return mutated_model

def averageCrossover(parentA, parentB, mutationRate=0.1):
    child = EvolutionNetwork(92, 256, 3)
    
    # State dict stores all the values of the network
    # Pull states of both parents
    parent1_state_dict = parentA.state_dict()
    parent2_state_dict = parentB.state_dict()
    
    # Get the state of the child
    child_state_dict = child.state_dict()
    
    # Child is based on both its parents
    for key in parent1_state_dict:
        child_state_dict[key] = (parent1_state_dict[key] + parent2_state_dict[key]) / 2
    
    # Load the averaged parameters into the child model
    child.load_state_dict(child_state_dict)

    # Add a bit of mutation to the child
    child = mutateModel(child, mutationRate=mutationRate)

    # Return new child
    return child

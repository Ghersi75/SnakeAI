import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import os
from helper import model_folder_path

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        if os.path.exists(os.path.join(model_folder_path, "model.pth")):
            model_state = torch.load(os.path.join(model_folder_path, "model.pth"))
            self.load_state_dict(model_state['model'])

    def forward(self, x):
        x = f.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, n_games, filename="model.pth"):
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)

        model_state = {
            'n_games': n_games,
            'model': self.state_dict()
        }

        filename = os.path.join(model_folder_path, filename)
        torch.save(model_state, filename)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # Only 1 number in form (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )

        # 1: Predict Q with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            # No idea lol
            target[idx][torch.argmax(action).item()] = Q_new

        # I have no idea whats going on here honestly, its a magic math function from a video
        # 2: Q_new = r + gamma * max(next_predicted Q) -> only if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new

        # zero grad empties the gradient, whatever that means
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
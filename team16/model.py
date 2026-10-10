import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


DEVICE = torch.device(
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


def _as_device_tensor(value, dtype, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)
        self.model_folder_path = './model'
        
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
    
    def save(self, file_name='model.pth'):
        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)
        
        file_name = os.path.join(self.model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        file_name = os.path.join(self.model_folder_path, file_name)
        state_dict = torch.load(
            file_name,
            map_location=next(self.parameters()).device,
            weights_only=True,
        )
        self.load_state_dict(state_dict)

class QTrainer:
    def __init__(self, model, lr, gamma, target_update_interval=100):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.target_model.eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)
        self.target_update_interval = target_update_interval
        self.training_steps = 0
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()
    
    def train_step(self, state, action, reward, next_state, game_over):
        device = next(self.model.parameters()).device
        states = _as_device_tensor(state, torch.float32, device)
        next_states = _as_device_tensor(next_state, torch.float32, device)
        actions = _as_device_tensor(action, torch.long, device).reshape(-1)
        rewards = _as_device_tensor(reward, torch.float32, device).reshape(-1)
        dones = _as_device_tensor(game_over, torch.bool, device).reshape(-1)

        if states.ndim == 1:
            states = states.unsqueeze(0)
            next_states = next_states.unsqueeze(0)
        if states.shape[0] != actions.shape[0] or any(
            tensor.shape[0] != states.shape[0]
            for tensor in (next_states, rewards, dones)
        ):
            raise ValueError(
                "States, actions, rewards, next states, and dones must have matching batch sizes"
            )

        self.model.train()
        predicted_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * next_q_values * (~dones)

        loss = self.criterion(predicted_q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_interval == 0:
            self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
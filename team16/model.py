import copy
import os
import tempfile
import uuid
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

# Trying to offload computation to gpu
def _as_device_tensor(value, dtype, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)

# Actual network model
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # I threw things together randoml. Is this sized correctly, optimal, effecive? I have no clue, probably not. 4 linear layers with the first three ReLU. 34 input, 256 hidden, 10 out
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)
        self.model_folder_path = './model'
        
    # Forward calculation with network. Takes state and outputs q values. Activation of ReLU, but last layer is just linear.
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

    # Save model so that the training is not wasted
    def save(self, file_name='model.pth'):
        os.makedirs(self.model_folder_path, exist_ok=True)
        target_path = os.path.join(self.model_folder_path, file_name)
        target_dir = os.path.dirname(target_path)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(file_name)}.",
            suffix=".tmp",
            dir=target_dir,
        )
        os.close(descriptor)

        # Was running into random issues and had to break-up saving into some temp states
        try:
            torch.save(self.state_dict(), temporary_path)
            try:
                os.replace(temporary_path, target_path)
            except OSError as error:
                if getattr(error, "winerror", None) not in (5, 1224):
                    raise
                checkpoint_path = os.path.join(
                    target_dir,
                    f"{os.path.basename(file_name)}.checkpoint-{uuid.uuid4().hex}.pth",
                )
                os.replace(temporary_path, checkpoint_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    # Load past training data and weights into the model
    def load(self, file_name='model.pth'):
        file_path = os.path.join(self.model_folder_path, file_name)
        checkpoint_prefix = f"{os.path.basename(file_name)}.checkpoint-"
        checkpoint_paths = [
            os.path.join(self.model_folder_path, candidate)
            for candidate in os.listdir(self.model_folder_path)
            if candidate.startswith(checkpoint_prefix) and candidate.endswith(".pth")
        ]
        available_paths = [
            path for path in [file_path, *checkpoint_paths] if os.path.isfile(path)
        ]
        if available_paths:
            file_path = max(available_paths, key=os.path.getmtime)

        state_dict = torch.load(
            file_path,
            map_location=next(self.parameters()).device,
            weights_only=True,
        )
        self.load_state_dict(state_dict)

# Training. I think the training isn't that good, but after messing around couldn't really find a good way to improve it.
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

    # Trains a step with forward, backpropagation, and weight shifts
    def train_step(self, state, action, reward, next_state, game_over):
        # Formats everything as needed and tries to make to make it fast, I think I wasted time here as the slowdown is the game and not this NN currently.
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

        # Go model, Go! I love chasing Targets. I think that with the values set and the models state that this was never too effective at actually training it as it was not weighing future states accurately
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

    # Trying to do the thing where the targe is not always moving, but I don't know how much it helped.
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RLAgent(nn.Module):
    """
    Reinforcement learning agent for optimizing irradiation schedules.
    Uses a simple policy gradient approach.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        logits = self.fc2(x)
        return torch.softmax(logits, dim=-1)

    def select_action(self, state):
        probs = self.forward(state)
        action = torch.multinomial(probs, 1)
        idx = action.item()
        return idx, probs[0][int(idx)].item()

def train_rl(env, agent, n_episodes=100, gamma=0.99, lr=1e-2):
    """
    Train the RL agent in the given environment.
    env must have reset(), step(action), and state_dim/action_dim attributes.
    """
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    for episode in range(n_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            action, prob = agent.select_action(state_tensor)
            next_state, reward, done, _ = env.step(action)
            log_probs.append(torch.log(torch.tensor(prob)))
            rewards.append(reward)
            state = next_state
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        # Policy gradient update
        loss = -torch.stack(log_probs) * (returns - returns.mean())
        loss = loss.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if episode % 10 == 0:
            print(f"Episode {episode}: Total reward = {sum(rewards):.2f}") 
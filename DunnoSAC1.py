import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import random

# Hyperparameters
BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
LEARNING_RATE = 0.0003
ALPHA = 0.2  # Temperature parameter for entropy

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Networks
        self.q1 = self._create_q_network(state_dim, action_dim)
        self.q2 = self._create_q_network(state_dim, action_dim)
        self.q1_target = self._create_q_network(state_dim, action_dim)
        self.q2_target = self._create_q_network(state_dim, action_dim)
        
        # Policy Network
        self.policy = self._create_policy_network(state_dim, action_dim)
        
        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=LEARNING_RATE)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=LEARNING_RATE)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        
        # Copy weights to target networks
        self._update_target_networks(1.0)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        
        # Action space
        self.action_dim = action_dim
    
    def _create_q_network(self, state_dim, action_dim):
        network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)
        return network
    
    def _create_policy_network(self, state_dim, action_dim):
        network = PolicyNetwork(state_dim, action_dim).to(self.device)
        return network
    
    def _update_target_networks(self, tau):
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.policy.sample(state)
        return action.cpu().numpy()[0]
    
    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return 0
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        # Compute Q-values
        q1_values = self.q1(torch.cat([states, actions], 1))
        q2_values = self.q2(torch.cat([states, actions], 1))
        
        # Sample next actions
        next_actions, log_probs = self.policy.sample(next_states)
        
        # Compute target Q-values
        next_q1 = self.q1_target(torch.cat([next_states, next_actions], 1))
        next_q2 = self.q2_target(torch.cat([next_states, next_actions], 1))
        next_q_values = torch.min(next_q1, next_q2) - ALPHA * log_probs
        target_q_values = rewards + (1 - dones) * GAMMA * next_q_values
        
        # Q-network loss
        q1_loss = F.mse_loss(q1_values, target_q_values.detach())
        q2_loss = F.mse_loss(q2_values, target_q_values.detach())
        
        # Update Q-networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Policy network loss
        new_actions, log_probs = self.policy.sample(states)
        q1_values = self.q1(torch.cat([states, new_actions], 1))
        q2_values = self.q2(torch.cat([states, new_actions], 1))
        min_q_values = torch.min(q1_values, q2_values)
        
        policy_loss = (ALPHA * log_probs - min_q_values).mean()
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Soft update of target networks
        self._update_target_networks(TAU)
        
        return policy_loss.item()

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
    
    def forward(self, state):
        x = self.network(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Reparameterization trick
        noise = torch.randn_like(mean)
        action = mean + std * noise
        
        # Compute log probability
        log_prob = -0.5 * ((noise / std) ** 2 + 2 * torch.log(std) + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Tanh squashing
        action = torch.tanh(action)
        
        return action, log_prob

def train_sac():
    # Create environment
    env = gym.make('MountainCarContinuous-v0')
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize agent
    agent = SACAgent(state_dim, action_dim)
    
    # Tracking variables
    episode_rewards = []
    successful_episodes = 0
    start_time = time.time()
    
    # Training loop
    for episode in range(100):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(1000):
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store in replay buffer
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # Train agent
            agent.train()
            
            state = next_state
            episode_reward += reward
            
            if done:
                # Check if episode was successful (reached the goal)
                if terminated:
                    successful_episodes += 1
                break
        
        # Store episode reward
        episode_rewards.append(episode_reward)
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")
    
    # Calculate training statistics
    training_time = time.time() - start_time
    avg_reward = np.mean(episode_rewards)
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    plt.plot(episode_rewards)
    plt.title('Episode Rewards during Training (SAC)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Add statistics text
    stats_text = (
        f"Training Time: {training_time:.2f} seconds\n"
        f"Successful Episodes: {successful_episodes}/100\n"
        f"Average Reward: {avg_reward:.2f}\n"
        f"Total Episodes: 100"
    )
    plt.text(0.65, 0.05, stats_text, 
             transform=plt.gca().transAxes, 
             verticalalignment='bottom', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('mountain_car_sac_rewards.png')
    plt.close()
    
    # Close environment
    env.close()
    
    return {
        'training_time': training_time,
        'successful_episodes': successful_episodes,
        'average_reward': avg_reward,
        'total_episodes': 100
    }

# Run training
if __name__ == "__main__":
    results = train_sac()
    print("\nTraining Results:")
    for key, value in results.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
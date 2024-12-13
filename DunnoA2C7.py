import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Normal
import time

# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99
NUM_EPISODES = 1000
MAX_STEPS = 1000
ENTROPY_WEIGHT = 0.01

class A2CNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(A2CNetwork, self).__init__()
        
        # Actor network (policy network)
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        # Critic network (value network)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        # Get mean and std for action distribution
        action_mean = self.actor(state)
        action_std = torch.exp(torch.clamp(self.actor(state), min=-5, max=2))
        
        # Get state value
        state_value = self.critic(state)
        
        return action_mean, action_std, state_value

class A2CAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # Create network and optimizer
        self.network = A2CNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        
        # Training tracking
        self.episode_rewards = []
        self.total_training_time = 0
        self.successful_episodes = 0
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action distribution parameters
        action_mean, action_std, _ = self.network(state)
        
        # Create normal distribution for continuous action space
        dist = Normal(action_mean, action_std)
        
        # Sample action
        action = dist.sample()
        
        # Log probability for gradient computation
        log_prob = dist.log_prob(action).sum()
        
        return action.detach().numpy()[0], log_prob
    
    def compute_returns(self, rewards, dones):
        # Compute discounted returns
        returns = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + GAMMA * R
            returns.insert(0, R)
        
        return torch.tensor(returns)
    
    def train(self):
        start_time = time.time()
        
        for episode in range(NUM_EPISODES):
            state, _ = self.env.reset()
            episode_reward = 0
            rewards = []
            log_probs = []
            values = []
            dones = []
            
            for step in range(MAX_STEPS):
                # Select action
                action, log_prob = self.select_action(state)
                
                # Take step in environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                # Get value of current state
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                _, _, value = self.network(state_tensor)
                
                # Store experiences
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                dones.append(terminated or truncated)
                
                episode_reward += reward
                state = next_state
                
                if terminated or truncated:
                    break
            
            # Compute returns and advantages
            returns = self.compute_returns(rewards, dones)
            values = torch.stack(values).squeeze()
            advantages = returns - values
            log_probs = torch.stack(log_probs)
            
            # Compute actor and critic losses
            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = advantages.pow(2).mean()
            
            # Total loss with entropy bonus for exploration
            loss = actor_loss + 0.5 * critic_loss
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track performance
            self.episode_rewards.append(episode_reward)
            
            if episode_reward >= -120:  # Close to solving the environment
                self.successful_episodes += 1
            
            # Print progress
            if episode % 10 == 0:
                print(f"Episode {episode}: Reward = {episode_reward}")
        
        # Compute total training time
        self.total_training_time = time.time() - start_time
    
    def plot_results(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot performance stats box
        plt.subplot(1, 2, 2)
        plt.axis('off')
        stats_text = (
            f"Training Statistics:\n"
            f"Total Training Time: {self.total_training_time:.2f} sec\n"
            f"Successful Episodes: {self.successful_episodes}\n"
            f"Average Reward: {np.mean(self.episode_rewards):.2f}\n"
            f"Total Episodes: {NUM_EPISODES}"
        )
        plt.text(0.5, 0.5, stats_text, 
                 horizontalalignment='center', 
                 verticalalignment='center', 
                 bbox=dict(facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        plt.show()

# Main execution
env = gym.make('MountainCarContinuous-v0')
agent = A2CAgent(env)
agent.train()
agent.plot_results()
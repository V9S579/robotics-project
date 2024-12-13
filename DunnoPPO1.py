import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import time

# Hyperparameters
LEARNING_RATE = 0.0003
GAMMA = 0.99
LAMBDA = 0.95
CLIP_RATIO = 0.2
EPOCHS = 3
BATCH_SIZE = 64
NUM_EPISODES = 500
MAX_STEPS = 1000

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor Network (Policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ).to(self.device)
        
        # Critic Network (Value Function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        
        # Mean and Std for action distribution
        self.log_std = nn.Parameter(torch.zeros(action_dim)).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()) + [self.log_std], lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)
    
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        mean = self.actor(state)
        std = torch.exp(self.log_std)
        
        # Create normal distribution
        dist = Normal(mean, std)
        action = dist.sample()
        
        # Log probability of the action
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.detach().cpu().numpy(), log_prob.detach().cpu().item()
    
    def compute_gae(self, rewards, values, dones):
        # Generalized Advantage Estimation
        advantages = []
        gae = 0
        
        # Ensure values has one more element for next state value
        values = np.append(values, 0)
        
        for step in reversed(range(len(rewards))):
            # Calculate TD error
            delta = rewards[step] + GAMMA * values[step + 1] * (1 - dones[step]) - values[step]
            
            # Calculate GAE
            gae = delta + GAMMA * LAMBDA * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages)
    
    def update(self, states, actions, old_log_probs, rewards, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        
        # Compute values
        values = self.critic(states).squeeze().detach().cpu().numpy()
        
        # Compute returns and advantages
        returns = np.zeros_like(rewards)
        advantages = self.compute_gae(rewards, values, dones)
        
        # Compute returns (GAE + values)
        for t in range(len(rewards)):
            returns[t] = advantages[t] + values[t]
        
        # Convert to tensors
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = advantages.to(self.device)
        
        # PPO Update
        for _ in range(EPOCHS):
            # Compute current policy distribution
            mean = self.actor(states)
            std = torch.exp(self.log_std)
            dist = Normal(mean, std)
            
            # Compute log probabilities of current actions
            log_probs = dist.log_prob(actions).sum(dim=-1)
            
            # Compute ratio
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Compute surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            critic_loss = nn.MSELoss()(self.critic(states).squeeze(), returns)
            
            # Update networks
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            (actor_loss + critic_loss).backward()
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()

def train_ppo():
    # Create environment
    env = gym.make('MountainCarContinuous-v0')
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize agent
    agent = PPOAgent(state_dim, action_dim)
    
    # Tracking variables
    episode_rewards = []
    successful_episodes = 0
    start_time = time.time()
    
    # Training loop
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        
        states, actions, rewards, dones, log_probs = [], [], [], [], []
        
        for step in range(MAX_STEPS):
            # Select action
            action, log_prob = agent.select_action(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            
            state = next_state
            episode_reward += reward
            
            if done:
                # Check if episode was successful (reached the goal)
                if terminated:
                    successful_episodes += 1
                break
        
        # Store episode reward
        episode_rewards.append(episode_reward)
        
        # Update agent only if we have enough experiences
        if len(states) > 1:
            agent.update(states, actions, log_probs, rewards, dones)
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")
    
    # Calculate training statistics
    training_time = time.time() - start_time
    avg_reward = np.mean(episode_rewards)
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    plt.plot(episode_rewards)
    plt.title('Episode Rewards during Training')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Add statistics text
    stats_text = (
        f"Training Time: {training_time:.2f} seconds\n"
        f"Successful Episodes: {successful_episodes}/{NUM_EPISODES}\n"
        f"Average Reward: {avg_reward:.2f}\n"
        f"Total Episodes: {NUM_EPISODES}"
    )
    plt.text(0.65, 0.05, stats_text, 
             transform=plt.gca().transAxes, 
             verticalalignment='bottom', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('mountain_car_ppo_rewards.png')
    plt.close()
    
    # Close environment
    env.close()
    
    return {
        'training_time': training_time,
        'successful_episodes': successful_episodes,
        'average_reward': avg_reward,
        'total_episodes': NUM_EPISODES
    }

# Run training
if __name__ == "__main__":
    results = train_ppo()
    print("\nTraining Results:")
    for key, value in results.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
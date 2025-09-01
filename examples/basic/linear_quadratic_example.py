#!/usr/bin/env python3
"""
Basic example of solving a Linear Quadratic Mean Field Game using RL.

This example demonstrates how to use the MFG_RL library to solve
a simple linear quadratic mean field game.
"""

import numpy as np
import matplotlib.pyplot as plt
from mfg_rl.environments import LinearQuadraticMFG
from mfg_rl.algorithms import MeanFieldDQN
from mfg_rl.utils import Config, Logger


def main():
    """Run the linear quadratic MFG example."""
    # Configuration
    config = Config({
        'state_dim': 2,
        'action_dim': 1, 
        'population_size': 100,
        'episode_length': 50,
        'learning_rate': 0.001,
        'batch_size': 32,
        'n_episodes': 1000
    })
    
    # Initialize environment
    env = LinearQuadraticMFG(config)
    
    # Initialize algorithm
    algorithm = MeanFieldDQN(config, env)
    
    # Initialize logger
    logger = Logger(config)
    
    # Training loop
    print("Starting training...")
    rewards = []
    
    for episode in range(config.n_episodes):
        episode_reward = algorithm.train_episode()
        rewards.append(episode_reward)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Average Reward: {np.mean(rewards[-100:]):.3f}")
            
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Training Progress - Linear Quadratic MFG')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig('results/plots/lq_training.png')
    plt.show()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
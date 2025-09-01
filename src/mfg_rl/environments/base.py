"""
Base environment class for Mean Field Games.

This module defines the abstract interface for MFG environments,
following the design principles from MFGLib.
"""

from typing import Dict, Any, Callable, Tuple, Protocol, Optional
import torch
import numpy as np
from abc import ABC, abstractmethod


class RewardFunction(Protocol):
    """Protocol for reward functions in Mean Field Games."""
    
    def __call__(self, 
                 state: torch.Tensor, 
                 action: torch.Tensor, 
                 mean_field: torch.Tensor,
                 time: Optional[float] = None) -> torch.Tensor:
        """
        Compute reward for given state, action, and mean field.
        
        Args:
            state: Agent state (batch_size, state_dim)
            action: Agent action (batch_size, action_dim)
            mean_field: Mean field distribution (batch_size, mf_dim)
            time: Current time (optional)
            
        Returns:
            Reward values (batch_size,)
        """
        ...


class TransitionFunction(Protocol):
    """Protocol for transition functions in Mean Field Games."""
    
    def __call__(self,
                 state: torch.Tensor,
                 action: torch.Tensor,
                 mean_field: torch.Tensor,
                 time: Optional[float] = None) -> torch.Tensor:
        """
        Compute next state given current state, action, and mean field.
        
        Args:
            state: Current state (batch_size, state_dim)
            action: Action taken (batch_size, action_dim)
            mean_field: Mean field distribution (batch_size, mf_dim)
            time: Current time (optional)
            
        Returns:
            Next state (batch_size, state_dim)
        """
        ...


class MFGEnvironment(ABC):
    """
    Abstract base class for Mean Field Game environments.
    
    This class defines the interface for MFG environments, emphasizing
    the mathematical structure of mean field games rather than step-based
    simulation like traditional RL environments.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 time_horizon: float,
                 dt: float = 0.01,
                 population_size: Optional[int] = None):
        """
        Initialize MFG environment.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            time_horizon: Time horizon for the game
            dt: Time discretization step
            population_size: Population size (for finite populations)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_horizon = time_horizon
        self.dt = dt
        self.population_size = population_size
        
        # Time grid
        self.time_grid = torch.arange(0, time_horizon + dt, dt)
        self.n_time_steps = len(self.time_grid)
    
    @abstractmethod
    def reward_function(self, 
                       state: torch.Tensor,
                       action: torch.Tensor, 
                       mean_field: torch.Tensor,
                       time: Optional[float] = None) -> torch.Tensor:
        """Compute reward function."""
        pass
    
    @abstractmethod
    def transition_function(self,
                           state: torch.Tensor,
                           action: torch.Tensor,
                           mean_field: torch.Tensor,
                           time: Optional[float] = None) -> torch.Tensor:
        """Compute state transition."""
        pass
    
    @abstractmethod
    def terminal_cost(self, state: torch.Tensor, mean_field: torch.Tensor) -> torch.Tensor:
        """Compute terminal cost at final time."""
        pass
    
    @abstractmethod
    def initial_distribution(self) -> torch.Tensor:
        """Get initial state distribution."""
        pass
    
    def sample_initial_states(self, n_samples: int) -> torch.Tensor:
        """Sample initial states from the initial distribution."""
        # Default implementation - subclasses should override for specific distributions
        return torch.randn(n_samples, self.state_dim)
    
    def state_space_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get state space bounds."""
        # Default: unbounded
        return (
            torch.full((self.state_dim,), -np.inf),
            torch.full((self.state_dim,), np.inf)
        )
    
    def action_space_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action space bounds."""
        # Default: unbounded
        return (
            torch.full((self.action_dim,), -np.inf),
            torch.full((self.action_dim,), np.inf)
        )
    
    def simulate_trajectory(self,
                          policy,
                          initial_state: torch.Tensor,
                          mean_field_trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate a single agent trajectory given policy and mean field.
        
        Args:
            policy: Agent policy
            initial_state: Initial state (state_dim,)
            mean_field_trajectory: Mean field over time (n_time_steps, mf_dim)
            
        Returns:
            Tuple of (states, actions, rewards) over time
        """
        states = torch.zeros(self.n_time_steps, self.state_dim)
        actions = torch.zeros(self.n_time_steps - 1, self.action_dim)  
        rewards = torch.zeros(self.n_time_steps - 1)
        
        states[0] = initial_state
        
        for t in range(self.n_time_steps - 1):
            current_state = states[t].unsqueeze(0)  # Add batch dimension
            current_mean_field = mean_field_trajectory[t].unsqueeze(0)
            
            # Get action from policy
            action = policy.sample_action(current_state, current_mean_field)
            actions[t] = action.squeeze(0)
            
            # Compute reward
            reward = self.reward_function(current_state, action.unsqueeze(0), 
                                        current_mean_field, self.time_grid[t])
            rewards[t] = reward.squeeze(0)
            
            # Compute next state
            next_state = self.transition_function(current_state, action.unsqueeze(0),
                                                current_mean_field, self.time_grid[t])
            states[t + 1] = next_state.squeeze(0)
        
        return states, actions, rewards
    
    def compute_value_function(self,
                             policy,
                             initial_state: torch.Tensor,
                             mean_field_trajectory: torch.Tensor) -> float:
        """
        Compute value function for given policy and mean field trajectory.
        
        Args:
            policy: Agent policy
            initial_state: Initial state
            mean_field_trajectory: Mean field trajectory over time
            
        Returns:
            Value function (total expected return)
        """
        states, actions, rewards = self.simulate_trajectory(
            policy, initial_state, mean_field_trajectory
        )
        
        # Add terminal cost
        terminal_state = states[-1].unsqueeze(0)
        final_mean_field = mean_field_trajectory[-1].unsqueeze(0)
        terminal_reward = self.terminal_cost(terminal_state, final_mean_field)
        
        # Compute total discounted return (assuming no discounting for now)
        total_return = rewards.sum() + terminal_reward.squeeze(0)
        return total_return.item()
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get environment parameters."""
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'time_horizon': self.time_horizon,
            'dt': self.dt,
            'population_size': self.population_size,
        }
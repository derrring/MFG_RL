"""
Policy representations and utilities for Mean Field Games.

This module provides policy classes and utilities specifically designed
for mean field games, including mean-field aware policies.
"""

from typing import Dict, Any, Optional, Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod


class MFGPolicy(ABC):
    """
    Abstract base class for Mean Field Game policies.
    
    Unlike standard RL policies, MFG policies take both the agent's state
    and the mean field (population distribution) as inputs.
    """
    
    @abstractmethod
    def forward(self, state: torch.Tensor, mean_field: torch.Tensor) -> torch.Tensor:
        """
        Compute policy output given state and mean field.
        
        Args:
            state: Agent state (batch_size, state_dim)
            mean_field: Mean field distribution (batch_size, mf_dim)
            
        Returns:
            Policy output (action probabilities or values)
        """
        pass
    
    @abstractmethod
    def sample_action(self, state: torch.Tensor, mean_field: torch.Tensor) -> torch.Tensor:
        """Sample action from the policy."""
        pass
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get policy parameters for serialization."""
        return {}


class NeuralMFGPolicy(MFGPolicy, nn.Module):
    """
    Neural network-based policy for Mean Field Games.
    
    This policy uses neural networks to map (state, mean_field) pairs
    to action distributions.
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int, 
                 mean_field_dim: int,
                 hidden_dims: List[int] = [64, 64],
                 activation: str = "relu",
                 policy_type: str = "stochastic"):
        """
        Initialize neural MFG policy.
        
        Args:
            state_dim: Dimension of agent state
            action_dim: Dimension of action space
            mean_field_dim: Dimension of mean field representation
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            policy_type: "stochastic" or "deterministic"
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mean_field_dim = mean_field_dim
        self.policy_type = policy_type
        
        # Input dimension is state + mean field
        input_dim = state_dim + mean_field_dim
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
            ])
            prev_dim = hidden_dim
        
        # Output layer
        if policy_type == "stochastic":
            layers.append(nn.Linear(prev_dim, action_dim))
        else:  # deterministic
            layers.extend([
                nn.Linear(prev_dim, action_dim),
                nn.Tanh()  # Bounded actions
            ])
            
        self.network = nn.Sequential(*layers)
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, state: torch.Tensor, mean_field: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the policy network.
        
        Args:
            state: Agent state (batch_size, state_dim)
            mean_field: Mean field (batch_size, mean_field_dim)
            
        Returns:
            Policy output (batch_size, action_dim)
        """
        # Concatenate state and mean field
        inputs = torch.cat([state, mean_field], dim=-1)
        outputs = self.network(inputs)
        
        if self.policy_type == "stochastic":
            # Apply softmax for action probabilities
            return F.softmax(outputs, dim=-1)
        else:
            return outputs
    
    def sample_action(self, state: torch.Tensor, mean_field: torch.Tensor) -> torch.Tensor:
        """Sample action from the policy."""
        policy_output = self.forward(state, mean_field)
        
        if self.policy_type == "stochastic":
            # Sample from categorical distribution
            dist = torch.distributions.Categorical(policy_output)
            return dist.sample()
        else:
            # Return deterministic action
            return policy_output
    
    def log_prob(self, state: torch.Tensor, mean_field: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of action."""
        policy_output = self.forward(state, mean_field)
        
        if self.policy_type == "stochastic":
            dist = torch.distributions.Categorical(policy_output)
            return dist.log_prob(action)
        else:
            # For deterministic policies, return 0 (delta function)
            return torch.zeros(action.shape[0])
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get policy parameters."""
        return {name: param.clone() for name, param in self.named_parameters()}


class LinearQuadraticPolicy(MFGPolicy):
    """
    Analytical policy for Linear Quadratic Mean Field Games.
    
    For LQ-MFG problems, the optimal policy has a known linear form
    that can be computed analytically.
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 Q: torch.Tensor,
                 R: torch.Tensor,
                 mean_field_coupling: float = 1.0):
        """
        Initialize LQ policy.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension  
            Q: State cost matrix
            R: Action cost matrix
            mean_field_coupling: Coupling strength with mean field
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.Q = Q
        self.R = R
        self.mean_field_coupling = mean_field_coupling
        
        # Policy parameters (to be learned or computed)
        self.K = torch.zeros(action_dim, state_dim)  # Feedback gain
        self.k = torch.zeros(action_dim)  # Mean field term
    
    def forward(self, state: torch.Tensor, mean_field: torch.Tensor) -> torch.Tensor:
        """
        Compute linear policy output.
        
        Args:
            state: Agent state
            mean_field: Mean field (scalar mean for LQ case)
            
        Returns:
            Action: K * state + k * mean_field
        """
        state_term = torch.matmul(state, self.K.T)
        mean_field_term = self.k * mean_field.mean(dim=-1, keepdim=True)
        return state_term + mean_field_term
    
    def sample_action(self, state: torch.Tensor, mean_field: torch.Tensor) -> torch.Tensor:
        """For deterministic policy, sample_action is the same as forward."""
        return self.forward(state, mean_field)
    
    def update_parameters(self, K: torch.Tensor, k: torch.Tensor):
        """Update policy parameters."""
        self.K = K
        self.k = k
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get policy parameters."""
        return {"K": self.K.clone(), "k": self.k.clone()}


class TabularMFGPolicy(MFGPolicy):
    """
    Tabular policy for discrete Mean Field Games.
    
    Maintains explicit action probabilities for each (state, mean_field) pair.
    """
    
    def __init__(self, 
                 num_states: int,
                 num_actions: int,
                 num_mean_field_bins: int = 10):
        """
        Initialize tabular MFG policy.
        
        Args:
            num_states: Number of discrete states
            num_actions: Number of discrete actions
            num_mean_field_bins: Number of mean field discretization bins
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_mean_field_bins = num_mean_field_bins
        
        # Policy table: (state, mean_field_bin, action)
        self.policy_table = torch.ones(num_states, num_mean_field_bins, num_actions)
        self.policy_table = self.policy_table / num_actions  # Uniform initialization
    
    def forward(self, state: torch.Tensor, mean_field: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities from policy table.
        
        Args:
            state: Discrete state indices
            mean_field: Mean field values (will be discretized)
            
        Returns:
            Action probabilities
        """
        # Discretize mean field
        mean_field_bins = self._discretize_mean_field(mean_field)
        
        # Look up policy table
        batch_size = state.shape[0]
        action_probs = torch.zeros(batch_size, self.num_actions)
        
        for i in range(batch_size):
            s = int(state[i])
            mf_bin = int(mean_field_bins[i])
            action_probs[i] = self.policy_table[s, mf_bin]
            
        return action_probs
    
    def sample_action(self, state: torch.Tensor, mean_field: torch.Tensor) -> torch.Tensor:
        """Sample action from tabular policy."""
        action_probs = self.forward(state, mean_field)
        dist = torch.distributions.Categorical(action_probs)
        return dist.sample()
    
    def update_policy(self, state: int, mean_field_bin: int, action_probs: torch.Tensor):
        """Update policy table entry."""
        self.policy_table[state, mean_field_bin] = action_probs
    
    def _discretize_mean_field(self, mean_field: torch.Tensor) -> torch.Tensor:
        """Discretize continuous mean field values into bins."""
        # Simple uniform discretization between 0 and 1
        bins = torch.linspace(0, 1, self.num_mean_field_bins + 1)
        discretized = torch.bucketize(mean_field, bins) - 1
        return torch.clamp(discretized, 0, self.num_mean_field_bins - 1)
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get policy parameters."""
        return {"policy_table": self.policy_table.clone()}


def create_policy(policy_type: str, **kwargs) -> MFGPolicy:
    """
    Factory function to create MFG policies.
    
    Args:
        policy_type: Type of policy ("neural", "linear_quadratic", "tabular")
        **kwargs: Policy-specific arguments
        
    Returns:
        MFG policy instance
    """
    if policy_type == "neural":
        return NeuralMFGPolicy(**kwargs)
    elif policy_type == "linear_quadratic":
        return LinearQuadraticPolicy(**kwargs)
    elif policy_type == "tabular":
        return TabularMFGPolicy(**kwargs)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def policy_gradient_loss(policy: MFGPolicy,
                        states: torch.Tensor,
                        mean_fields: torch.Tensor, 
                        actions: torch.Tensor,
                        advantages: torch.Tensor) -> torch.Tensor:
    """
    Compute policy gradient loss for MFG policy.
    
    Args:
        policy: MFG policy
        states: Batch of states
        mean_fields: Batch of mean fields
        actions: Batch of actions taken
        advantages: Batch of advantage values
        
    Returns:
        Policy gradient loss
    """
    if hasattr(policy, 'log_prob'):
        log_probs = policy.log_prob(states, mean_fields, actions)
        return -(log_probs * advantages).mean()
    else:
        raise NotImplementedError("Policy does not support log_prob calculation")


from typing import List
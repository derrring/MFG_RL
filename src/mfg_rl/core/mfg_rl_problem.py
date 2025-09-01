"""
Core MFG problem definition adapted for Reinforcement Learning.

This module defines the fundamental structure for Mean Field Games problems
that will be solved using RL methods, following the design patterns from MFG_PDE.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union
import torch
import numpy as np
from numpy.typing import NDArray


@dataclass
class MFGRLComponents:
    """
    Container for all components that define a custom MFG-RL problem.
    
    Similar to MFGComponents in MFG_PDE but adapted for RL approaches.
    This class holds all the mathematical and RL-specific components needed 
    to fully specify an MFG problem for RL solution methods.
    """
    
    # Core RL components
    reward_func: Optional[Callable] = None          # R(s, a, m, t) -> float
    transition_func: Optional[Callable] = None      # T(s, a, m, t) -> s'
    value_func: Optional[Callable] = None           # V(s, m, t) -> float
    policy_func: Optional[Callable] = None          # π(s, m, t) -> a
    
    # Mean field dynamics
    mean_field_func: Optional[Callable] = None      # Mean field evolution
    population_dynamics: Optional[Callable] = None  # Population state evolution
    
    # Initial conditions
    initial_state_dist: Optional[Callable] = None   # Initial state distribution
    initial_policy: Optional[Callable] = None       # Initial policy (if needed)
    initial_value: Optional[Callable] = None        # Initial value function
    
    # Terminal conditions  
    terminal_reward: Optional[Callable] = None      # Terminal reward function
    terminal_value: Optional[Callable] = None       # Terminal value function
    
    # Environment parameters
    state_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    action_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    time_horizon: float = 1.0
    dt: float = 0.01
    
    # RL-specific parameters
    discount_factor: float = 0.99
    exploration_noise: float = 0.1
    population_size: Optional[int] = None
    
    # Coupling parameters
    coupling_strength: float = 1.0                  # Mean field coupling strength
    interaction_kernel: Optional[Callable] = None   # Agent interaction kernel
    
    # Advanced RL components
    advantage_func: Optional[Callable] = None       # Advantage function
    baseline_func: Optional[Callable] = None        # Baseline for variance reduction
    
    # Meta-learning components (for adaptive algorithms)
    meta_policy: Optional[Callable] = None          # Meta-policy for adaptation
    adaptation_rule: Optional[Callable] = None     # How to adapt to mean field
    
    def validate(self) -> None:
        """Validate that essential components are provided."""
        required = ['reward_func', 'transition_func']
        missing = [comp for comp in required if getattr(self, comp) is None]
        if missing:
            raise ValueError(f"Missing required components: {missing}")


class MFGRLProblem(ABC):
    """
    Abstract base class for Mean Field Game problems solved with RL.
    
    This class follows the design patterns from MFG_PDE but adapts them
    for reinforcement learning solution methods.
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 time_horizon: float = 1.0,
                 dt: float = 0.01,
                 population_size: Optional[int] = None,
                 components: Optional[MFGRLComponents] = None):
        """
        Initialize MFG-RL problem.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            time_horizon: Time horizon for the game
            dt: Time discretization step
            population_size: Size of agent population
            components: Problem components container
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_horizon = time_horizon
        self.dt = dt
        self.population_size = population_size
        self.components = components or MFGRLComponents()
        
        # Derived properties
        self.n_time_steps = int(time_horizon / dt) + 1
        self.time_grid = torch.linspace(0, time_horizon, self.n_time_steps)
        
        # Validate components
        self.components.validate()
    
    @abstractmethod
    def reward(self, 
               state: torch.Tensor, 
               action: torch.Tensor, 
               mean_field: torch.Tensor,
               time: Optional[float] = None) -> torch.Tensor:
        """Compute reward function."""
        pass
    
    @abstractmethod 
    def transition(self,
                  state: torch.Tensor,
                  action: torch.Tensor, 
                  mean_field: torch.Tensor,
                  time: Optional[float] = None) -> torch.Tensor:
        """Compute state transition."""
        pass
    
    @abstractmethod
    def terminal_cost(self, 
                     state: torch.Tensor, 
                     mean_field: torch.Tensor) -> torch.Tensor:
        """Compute terminal cost."""
        pass
    
    @abstractmethod
    def initial_state_distribution(self) -> torch.distributions.Distribution:
        """Get initial state distribution."""
        pass
    
    def sample_initial_states(self, n_samples: int) -> torch.Tensor:
        """Sample initial states."""
        dist = self.initial_state_distribution()
        return dist.sample((n_samples,))
    
    def state_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get state space bounds."""
        if self.components.state_bounds:
            return self.components.state_bounds
        return (
            torch.full((self.state_dim,), -float('inf')),
            torch.full((self.state_dim,), float('inf'))
        )
    
    def action_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action space bounds."""
        if self.components.action_bounds:
            return self.components.action_bounds
        return (
            torch.full((self.action_dim,), -float('inf')),
            torch.full((self.action_dim,), float('inf'))
        )
    
    def compute_mean_field_evolution(self, 
                                   policies: Dict[str, Any],
                                   initial_distribution: torch.Tensor) -> torch.Tensor:
        """
        Compute mean field evolution given policies.
        
        Args:
            policies: Dictionary of agent policies
            initial_distribution: Initial state distribution
            
        Returns:
            Mean field trajectory over time
        """
        if self.components.mean_field_func:
            return self.components.mean_field_func(policies, initial_distribution)
        else:
            # Default: use empirical mean field computation
            from ..mean_field import EmpiricalMeanField
            computer = EmpiricalMeanField(
                num_samples=self.population_size or 1000,
                time_horizon=self.n_time_steps
            )
            return computer.compute(policies, initial_distribution)
    
    def get_problem_info(self) -> Dict[str, Any]:
        """Get problem information dictionary."""
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'time_horizon': self.time_horizon,
            'dt': self.dt,
            'n_time_steps': self.n_time_steps,
            'population_size': self.population_size,
            'discount_factor': self.components.discount_factor,
            'coupling_strength': self.components.coupling_strength,
        }


class LinearQuadraticMFGRL(MFGRLProblem):
    """
    Linear Quadratic Mean Field Game for RL.
    
    A concrete implementation of the classic LQ-MFG problem
    suitable for RL solution methods.
    """
    
    def __init__(self,
                 state_dim: int = 2,
                 action_dim: int = 1,
                 Q: Optional[torch.Tensor] = None,
                 R: Optional[torch.Tensor] = None,
                 A: Optional[torch.Tensor] = None,
                 B: Optional[torch.Tensor] = None,
                 sigma: float = 0.1,
                 coupling_strength: float = 1.0,
                 **kwargs):
        """
        Initialize Linear Quadratic MFG-RL problem.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension  
            Q: State cost matrix
            R: Action cost matrix
            A: State dynamics matrix
            B: Control dynamics matrix
            sigma: Noise standard deviation
            coupling_strength: Mean field coupling strength
        """
        
        # Default matrices if not provided
        if Q is None:
            Q = torch.eye(state_dim)
        if R is None:
            R = torch.eye(action_dim)
        if A is None:
            A = torch.eye(state_dim)
        if B is None:
            B = torch.ones(state_dim, action_dim)
            
        self.Q = Q
        self.R = R 
        self.A = A
        self.B = B
        self.sigma = sigma
        
        # Create components
        components = MFGRLComponents(
            coupling_strength=coupling_strength,
            exploration_noise=sigma,
        )
        
        super().__init__(state_dim, action_dim, components=components, **kwargs)
    
    def reward(self, state, action, mean_field, time=None):
        """LQ reward function."""
        state_cost = -0.5 * torch.sum(state * (self.Q @ state.T).T, dim=1)
        action_cost = -0.5 * torch.sum(action * (self.R @ action.T).T, dim=1)
        
        # Mean field coupling term
        mean_state = mean_field.mean(dim=0) if mean_field.dim() > 1 else mean_field
        coupling_cost = -self.components.coupling_strength * torch.sum(
            (state - mean_state.unsqueeze(0)) ** 2, dim=1
        )
        
        return state_cost + action_cost + coupling_cost
    
    def transition(self, state, action, mean_field, time=None):
        """LQ transition function."""
        # Deterministic dynamics: x' = Ax + Ba + noise  
        next_state = state @ self.A.T + action @ self.B.T
        
        # Add Gaussian noise
        if self.sigma > 0:
            noise = torch.randn_like(next_state) * self.sigma
            next_state = next_state + noise
            
        return next_state
    
    def terminal_cost(self, state, mean_field):
        """LQ terminal cost."""
        return -0.5 * torch.sum(state * (self.Q @ state.T).T, dim=1)
    
    def initial_state_distribution(self):
        """Gaussian initial distribution."""
        return torch.distributions.MultivariateNormal(
            torch.zeros(self.state_dim),
            torch.eye(self.state_dim)
        )


def create_mfg_rl_problem(problem_type: str, **kwargs) -> MFGRLProblem:
    """
    Factory function to create MFG-RL problems.
    
    Args:
        problem_type: Type of problem ("linear_quadratic", "crowd_motion", etc.)
        **kwargs: Problem-specific parameters
        
    Returns:
        MFG-RL problem instance
    """
    if problem_type == "linear_quadratic":
        return LinearQuadraticMFGRL(**kwargs)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


# Builder pattern for complex problem construction
class MFGRLProblemBuilder:
    """
    Builder for constructing complex MFG-RL problems.
    
    Following the builder pattern from MFG_PDE for flexible problem construction.
    """
    
    def __init__(self):
        self.components = MFGRLComponents()
        self._state_dim = None
        self._action_dim = None
        
    def with_reward_function(self, reward_func: Callable):
        """Add reward function."""
        self.components.reward_func = reward_func
        return self
        
    def with_transition_function(self, transition_func: Callable):
        """Add transition function."""
        self.components.transition_func = transition_func
        return self
        
    def with_dimensions(self, state_dim: int, action_dim: int):
        """Set state and action dimensions."""
        self._state_dim = state_dim
        self._action_dim = action_dim
        return self
        
    def with_time_horizon(self, time_horizon: float, dt: float = 0.01):
        """Set time parameters."""
        self.components.time_horizon = time_horizon
        self.components.dt = dt
        return self
        
    def with_population_size(self, population_size: int):
        """Set population size."""
        self.components.population_size = population_size
        return self
        
    def build(self) -> MFGRLProblem:
        """Build the problem."""
        if self._state_dim is None or self._action_dim is None:
            raise ValueError("State and action dimensions must be specified")
            
        # Create a generic problem with the specified components
        class CustomMFGRLProblem(MFGRLProblem):
            def reward(self, state, action, mean_field, time=None):
                return self.components.reward_func(state, action, mean_field, time)
                
            def transition(self, state, action, mean_field, time=None):
                return self.components.transition_func(state, action, mean_field, time)
                
            def terminal_cost(self, state, mean_field):
                if self.components.terminal_reward:
                    return self.components.terminal_reward(state, mean_field)
                return torch.zeros(state.shape[0])
                
            def initial_state_distribution(self):
                if self.components.initial_state_dist:
                    return self.components.initial_state_dist()
                return torch.distributions.MultivariateNormal(
                    torch.zeros(self.state_dim),
                    torch.eye(self.state_dim)
                )
        
        return CustomMFGRLProblem(
            self._state_dim,
            self._action_dim,
            time_horizon=self.components.time_horizon,
            dt=self.components.dt,
            population_size=self.components.population_size,
            components=self.components
        )
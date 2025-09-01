"""
Solver factory for creating MFG-RL algorithms with intelligent defaults.

This module follows the factory pattern from MFG_PDE, providing
intelligent solver creation with automatic backend selection and
optimal configuration based on problem characteristics.
"""

from typing import Dict, Any, Optional, Union, Type
import torch
from dataclasses import dataclass
from enum import Enum

from ..core.mfg_rl_problem import MFGRLProblem
from ..algorithms import MeanFieldDQN, MeanFieldPolicyGradient, MeanFieldActorCritic
from ..config import MFGRLConfig


class SolverType(Enum):
    """Available solver types."""
    MEAN_FIELD_DQN = "mean_field_dqn"
    MEAN_FIELD_POLICY_GRADIENT = "mean_field_pg" 
    MEAN_FIELD_ACTOR_CRITIC = "mean_field_ac"
    AUTO = "auto"


class BackendType(Enum):
    """Available backend types."""
    PYTORCH = "pytorch"
    JAX = "jax"
    TENSORFLOW = "tensorflow"
    AUTO = "auto"


@dataclass
class SolverCreationResult:
    """Result of solver creation with metadata."""
    solver: Any
    config: MFGRLConfig
    backend: str
    creation_time: float
    memory_estimate_mb: float
    recommended_batch_size: int


class SolverFactory:
    """
    Factory for creating MFG-RL solvers with intelligent defaults.
    
    This factory follows the MFG_PDE pattern of intelligent configuration
    and automatic optimization based on problem characteristics.
    """
    
    # Solver registry
    _solvers: Dict[SolverType, Type] = {
        SolverType.MEAN_FIELD_DQN: MeanFieldDQN,
        SolverType.MEAN_FIELD_POLICY_GRADIENT: MeanFieldPolicyGradient,
        SolverType.MEAN_FIELD_ACTOR_CRITIC: MeanFieldActorCritic,
    }
    
    @classmethod
    def create_solver(cls,
                     problem: MFGRLProblem,
                     solver_type: Union[str, SolverType] = SolverType.AUTO,
                     backend: Union[str, BackendType] = BackendType.AUTO,
                     config: Optional[MFGRLConfig] = None,
                     **kwargs) -> SolverCreationResult:
        """
        Create an MFG-RL solver with intelligent defaults.
        
        Args:
            problem: MFG-RL problem to solve
            solver_type: Type of solver to create
            backend: Backend to use (pytorch/jax/tensorflow)
            config: Optional configuration (will create optimal if None)
            **kwargs: Additional solver-specific parameters
            
        Returns:
            SolverCreationResult with solver and metadata
        """
        import time
        start_time = time.time()
        
        # Convert string enums
        if isinstance(solver_type, str):
            solver_type = SolverType(solver_type)
        if isinstance(backend, str):
            backend = BackendType(backend)
            
        # Auto-select solver type
        if solver_type == SolverType.AUTO:
            solver_type = cls._auto_select_solver_type(problem)
            
        # Auto-select backend
        if backend == BackendType.AUTO:
            backend = cls._auto_select_backend(problem)
            
        # Create or optimize config
        if config is None:
            config = cls._create_optimal_config(problem, solver_type, backend)
        else:
            config = cls._optimize_config(config, problem, solver_type)
            
        # Create solver
        solver_class = cls._solvers[solver_type]
        solver = solver_class(problem=problem, config=config, **kwargs)
        
        # Compute metadata
        creation_time = time.time() - start_time
        memory_estimate = cls._estimate_memory_usage(problem, config)
        batch_size = cls._recommend_batch_size(problem, config)
        
        return SolverCreationResult(
            solver=solver,
            config=config,
            backend=backend.value,
            creation_time=creation_time,
            memory_estimate_mb=memory_estimate,
            recommended_batch_size=batch_size
        )
    
    @classmethod
    def _auto_select_solver_type(cls, problem: MFGRLProblem) -> SolverType:
        """Automatically select optimal solver type based on problem characteristics."""
        problem_info = problem.get_problem_info()
        
        # Simple heuristics - could be made more sophisticated
        if problem_info['action_dim'] == 1 and problem_info['state_dim'] <= 10:
            return SolverType.MEAN_FIELD_DQN
        elif problem_info['state_dim'] <= 5:
            return SolverType.MEAN_FIELD_ACTOR_CRITIC  
        else:
            return SolverType.MEAN_FIELD_POLICY_GRADIENT
    
    @classmethod  
    def _auto_select_backend(cls, problem: MFGRLProblem) -> BackendType:
        """Automatically select optimal backend."""
        # Check CUDA availability
        if torch.cuda.is_available():
            return BackendType.PYTORCH
        
        # Check problem size for JAX benefits
        problem_info = problem.get_problem_info()
        problem_size = problem_info['state_dim'] * problem_info.get('population_size', 1000)
        
        if problem_size > 10000:
            try:
                import jax
                return BackendType.JAX
            except ImportError:
                pass
                
        return BackendType.PYTORCH
    
    @classmethod
    def _create_optimal_config(cls,
                              problem: MFGRLProblem,
                              solver_type: SolverType,
                              backend: BackendType) -> MFGRLConfig:
        """Create optimal configuration for problem and solver combination."""
        from ..config import create_fast_config, create_accurate_config
        
        problem_info = problem.get_problem_info()
        
        # Determine if we need accuracy vs speed
        if problem_info['state_dim'] > 10 or problem_info.get('population_size', 1000) > 5000:
            config = create_fast_config()
        else:
            config = create_accurate_config()
            
        # Solver-specific adjustments
        if solver_type == SolverType.MEAN_FIELD_DQN:
            config.algorithm.learning_rate *= 0.5  # DQN often needs lower LR
            config.algorithm.batch_size = min(config.algorithm.batch_size, 64)
        elif solver_type == SolverType.MEAN_FIELD_POLICY_GRADIENT:
            config.algorithm.batch_size *= 2  # PG benefits from larger batches
            
        # Backend-specific adjustments
        if backend == BackendType.JAX:
            config.algorithm.batch_size = max(config.algorithm.batch_size, 32)  # JAX prefers larger batches
            
        return config
    
    @classmethod
    def _optimize_config(cls,
                        config: MFGRLConfig,
                        problem: MFGRLProblem,
                        solver_type: SolverType) -> MFGRLConfig:
        """Optimize existing configuration for problem characteristics."""
        problem_info = problem.get_problem_info()
        
        # Adjust learning rate based on problem complexity
        if problem_info['state_dim'] > 10:
            config.algorithm.learning_rate *= 0.8
            
        # Adjust exploration based on action space
        if hasattr(config.algorithm, 'epsilon_decay'):
            if problem_info['action_dim'] > 5:
                config.algorithm.epsilon_decay = 0.999  # Slower decay for complex action spaces
                
        return config
    
    @classmethod
    def _estimate_memory_usage(cls, problem: MFGRLProblem, config: MFGRLConfig) -> float:
        """Estimate memory usage in MB."""
        problem_info = problem.get_problem_info()
        
        # Base memory for problem
        base_memory = (problem_info['state_dim'] + problem_info['action_dim']) * 4 / 1e6  # 4 bytes per float
        
        # Memory for networks (rough estimate)
        network_params = sum(config.network.hidden_layers) * problem_info['state_dim'] * 4 / 1e6
        
        # Memory for replay buffer
        buffer_memory = config.algorithm.memory_size * (problem_info['state_dim'] + problem_info['action_dim'] + 3) * 4 / 1e6
        
        return base_memory + network_params + buffer_memory
    
    @classmethod
    def _recommend_batch_size(cls, problem: MFGRLProblem, config: MFGRLConfig) -> int:
        """Recommend optimal batch size."""
        problem_info = problem.get_problem_info()
        
        # Base batch size from config
        base_batch_size = config.algorithm.batch_size
        
        # Adjust based on problem characteristics
        if problem_info['state_dim'] > 20:
            return max(base_batch_size // 2, 16)
        elif problem_info['state_dim'] < 5:
            return min(base_batch_size * 2, 256)
            
        return base_batch_size


# Convenience functions following MFG_PDE pattern
def create_fast_solver(problem: MFGRLProblem, **kwargs) -> Any:
    """
    Create a fast solver optimized for speed over accuracy.
    
    Args:
        problem: MFG-RL problem
        **kwargs: Additional parameters
        
    Returns:
        Configured solver optimized for speed
    """
    from ..config import create_fast_config
    
    config = create_fast_config()
    result = SolverFactory.create_solver(problem, config=config, **kwargs)
    return result.solver


def create_accurate_solver(problem: MFGRLProblem, **kwargs) -> Any:
    """
    Create an accurate solver optimized for precision over speed.
    
    Args:
        problem: MFG-RL problem  
        **kwargs: Additional parameters
        
    Returns:
        Configured solver optimized for accuracy
    """
    from ..config import create_accurate_config
    
    config = create_accurate_config()
    result = SolverFactory.create_solver(problem, config=config, **kwargs)
    return result.solver


def create_research_solver(problem: MFGRLProblem, **kwargs) -> Any:
    """
    Create a research solver with comprehensive logging and analysis.
    
    Args:
        problem: MFG-RL problem
        **kwargs: Additional parameters
        
    Returns:
        Configured solver optimized for research
    """
    from ..config import create_research_config
    
    config = create_research_config()
    result = SolverFactory.create_solver(problem, config=config, **kwargs)
    return result.solver


def create_solver(problem: MFGRLProblem, 
                 solver_type: str = "auto",
                 backend: str = "auto",
                 **kwargs) -> Any:
    """
    General solver creation function with auto-optimization.
    
    Args:
        problem: MFG-RL problem
        solver_type: Type of solver ("auto", "dqn", "pg", "ac")
        backend: Backend to use ("auto", "pytorch", "jax", "tensorflow")
        **kwargs: Additional parameters
        
    Returns:
        Optimally configured solver
    """
    result = SolverFactory.create_solver(problem, solver_type, backend, **kwargs)
    return result.solver
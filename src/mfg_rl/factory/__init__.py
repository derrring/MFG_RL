"""
Factory pattern for creating MFG-RL solvers and components.

This module provides factory functions for creating RL algorithms and solvers
optimized for Mean Field Games, following the design patterns from MFG_PDE.
"""

from .solver_factory import (
    SolverFactory,
    create_fast_solver,
    create_accurate_solver, 
    create_research_solver,
    create_solver
)

from .config_factory import (
    create_fast_config,
    create_accurate_config,
    create_research_config,
    create_custom_config
)

from .backend_factory import (
    BackendFactory,
    create_backend,
    get_available_backends,
    get_optimal_backend
)

__all__ = [
    # Solver factory
    "SolverFactory",
    "create_fast_solver", 
    "create_accurate_solver",
    "create_research_solver", 
    "create_solver",
    
    # Config factory
    "create_fast_config",
    "create_accurate_config", 
    "create_research_config",
    "create_custom_config",
    
    # Backend factory
    "BackendFactory",
    "create_backend",
    "get_available_backends",
    "get_optimal_backend"
]
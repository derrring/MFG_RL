"""
Configuration management for MFG-RL.

This module provides professional configuration management following
the patterns from MFG_PDE, with Pydantic validation and intelligent defaults.
"""

from .mfg_rl_config import (
    MFGRLConfig,
    AlgorithmConfig,
    NetworkConfig,
    TrainingConfig,
    LoggingConfig
)

from .config_factory import (
    create_fast_config,
    create_accurate_config,
    create_research_config,
    create_custom_config
)

from .config_validation import (
    validate_config,
    validate_compatibility,
    ConfigValidationError
)

__all__ = [
    # Core config classes
    "MFGRLConfig",
    "AlgorithmConfig", 
    "NetworkConfig",
    "TrainingConfig",
    "LoggingConfig",
    
    # Factory functions
    "create_fast_config",
    "create_accurate_config",
    "create_research_config", 
    "create_custom_config",
    
    # Validation
    "validate_config",
    "validate_compatibility",
    "ConfigValidationError"
]
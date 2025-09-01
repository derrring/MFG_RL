"""
Configuration factory functions following MFG_PDE patterns.

This module provides intelligent configuration creation with
optimized defaults for different use cases.
"""

from typing import Dict, Any, Optional
from .mfg_rl_config import (
    MFGRLConfig, 
    AlgorithmConfig, 
    NetworkConfig, 
    TrainingConfig, 
    LoggingConfig,
    ActivationFunction,
    OptimizerType,
    SchedulerType
)


def create_fast_config(**overrides) -> MFGRLConfig:
    """
    Create configuration optimized for speed over accuracy.
    
    This configuration prioritizes fast training and convergence,
    following the MFG_PDE pattern of intelligent defaults.
    
    Args:
        **overrides: Configuration overrides
        
    Returns:
        Fast-optimized MFG-RL configuration
    """
    # Algorithm optimized for speed
    algorithm = AlgorithmConfig(
        learning_rate=3e-3,  # Higher learning rate for faster convergence
        batch_size=64,       # Larger batches for efficiency
        memory_size=5000,    # Smaller replay buffer
        target_update_frequency=50,  # More frequent updates
        
        # Faster exploration decay
        epsilon_start=0.8,
        epsilon_end=0.05,
        epsilon_decay=0.98,
        
        # Aggressive optimization
        optimizer=OptimizerType.ADAM,
        scheduler=SchedulerType.EXPONENTIAL,
        grad_clip_norm=1.0,
        
        # Mean field updates
        mean_field_update_frequency=5,  # More frequent MF updates
        population_sample_size=500,     # Smaller population samples
    )
    
    # Simpler network for speed
    network = NetworkConfig(
        hidden_layers=[32, 32],  # Smaller networks
        activation=ActivationFunction.RELU,
        dropout=0.1,             # Light regularization
        batch_norm=False,        # Skip batch norm for speed
        layer_norm=False,
        initialization="he_uniform"
    )
    
    # Shorter training for quick results
    training = TrainingConfig(
        max_episodes=500,        # Fewer episodes
        max_steps_per_episode=100,
        eval_frequency=50,
        save_frequency=250,
        
        # Relaxed convergence
        convergence_threshold=1e-3,
        convergence_window=50,
        early_stopping_patience=25,
        
        # Parallel processing
        num_workers=2,
        use_multiprocessing=True
    )
    
    # Minimal logging for speed
    logging = LoggingConfig(
        console_level="WARNING",  # Reduce console output
        use_tensorboard=True,
        use_wandb=False,         # Skip W&B for speed
        log_frequency=20,
        save_trajectories=False,  # Skip trajectory saving
        save_mean_field=True,
        plot_frequency=200,
        save_plots=False         # Skip plot saving
    )
    
    config = MFGRLConfig(
        algorithm=algorithm,
        network=network,
        training=training,
        logging=logging,
        seed=42,
        device="auto",
        precision="float32"
    )
    
    # Apply any user overrides
    if overrides:
        config = config.update_from_dict(overrides)
        
    return config


def create_accurate_config(**overrides) -> MFGRLConfig:
    """
    Create configuration optimized for accuracy over speed.
    
    This configuration prioritizes solution quality and numerical precision,
    following the MFG_PDE pattern for research-grade results.
    
    Args:
        **overrides: Configuration overrides
        
    Returns:
        Accuracy-optimized MFG-RL configuration  
    """
    # Algorithm optimized for accuracy
    algorithm = AlgorithmConfig(
        learning_rate=1e-4,      # Lower learning rate for stability
        batch_size=32,           # Smaller batches for better gradients
        memory_size=50000,       # Large replay buffer
        target_update_frequency=1000,  # Slower, more stable updates
        
        # Careful exploration
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995,    # Very slow decay
        
        # Conservative optimization
        optimizer=OptimizerType.ADAMW,
        scheduler=SchedulerType.COSINE,
        grad_clip_norm=0.5,      # Careful gradient clipping
        
        # Precise mean field computation
        mean_field_update_frequency=20,
        population_sample_size=2000,   # Large population samples
    )
    
    # Larger network for better approximation
    network = NetworkConfig(
        hidden_layers=[128, 128, 64],  # Deeper network
        activation=ActivationFunction.GELU,
        dropout=0.2,                   # More regularization
        batch_norm=True,               # Batch normalization
        layer_norm=True,               # Layer normalization
        initialization="xavier_uniform"
    )
    
    # Extensive training for accuracy
    training = TrainingConfig(
        max_episodes=2000,             # More episodes
        max_steps_per_episode=300,
        eval_frequency=200,
        save_frequency=1000,
        
        # Strict convergence criteria
        convergence_threshold=1e-5,
        convergence_window=200,
        early_stopping_patience=100,
        
        # Single-threaded for determinism
        num_workers=1,
        use_multiprocessing=False
    )
    
    # Comprehensive logging for analysis
    logging = LoggingConfig(
        console_level="INFO",
        file_level="DEBUG",
        use_tensorboard=True,
        use_wandb=True,              # Enable W&B for research
        wandb_project="mfg_rl_accurate",
        log_frequency=5,             # Frequent logging
        save_trajectories=True,      # Save trajectories
        save_mean_field=True,
        plot_frequency=50,
        save_plots=True
    )
    
    config = MFGRLConfig(
        algorithm=algorithm,
        network=network,
        training=training,
        logging=logging,
        seed=42,
        device="auto",
        precision="float64"  # Double precision for accuracy
    )
    
    # Apply any user overrides
    if overrides:
        config = config.update_from_dict(overrides)
        
    return config


def create_research_config(**overrides) -> MFGRLConfig:
    """
    Create configuration optimized for research with comprehensive analysis.
    
    This configuration includes extensive logging, analysis, and reproducibility
    features for research purposes.
    
    Args:
        **overrides: Configuration overrides
        
    Returns:
        Research-optimized MFG-RL configuration
    """
    # Balanced algorithm settings
    algorithm = AlgorithmConfig(
        learning_rate=1e-3,
        batch_size=64,
        memory_size=20000,
        target_update_frequency=200,
        
        # Moderate exploration
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=0.999,
        
        # Research-friendly optimization
        optimizer=OptimizerType.ADAM,
        scheduler=SchedulerType.PLATEAU,
        grad_clip_norm=1.0,
        
        # Detailed mean field analysis
        mean_field_update_frequency=10,
        population_sample_size=1500,
    )
    
    # Moderate network for balance
    network = NetworkConfig(
        hidden_layers=[64, 64, 32],
        activation=ActivationFunction.RELU,
        dropout=0.15,
        batch_norm=True,
        layer_norm=False,
        initialization="xavier_uniform"
    )
    
    # Research-oriented training
    training = TrainingConfig(
        max_episodes=1500,
        max_steps_per_episode=250,
        eval_frequency=100,
        save_frequency=500,
        
        # Reasonable convergence
        convergence_threshold=5e-4,
        convergence_window=100,
        early_stopping_patience=75,
        
        # Flexible parallelization
        num_workers=2,
        use_multiprocessing=True
    )
    
    # Comprehensive research logging
    logging = LoggingConfig(
        console_level="INFO",
        file_level="DEBUG",
        experiment_name="mfg_rl_research",
        
        # Full logging suite
        use_tensorboard=True,
        use_wandb=True,
        wandb_project="mfg_rl_research",
        
        # Detailed metrics
        log_frequency=1,             # Log every episode
        save_trajectories=True,
        save_mean_field=True,
        
        # Rich visualizations
        plot_frequency=25,
        save_plots=True,
        plot_format="pdf"            # Publication-quality plots
    )
    
    config = MFGRLConfig(
        algorithm=algorithm,
        network=network,
        training=training,
        logging=logging,
        seed=42,
        device="auto",
        precision="float32"
    )
    
    # Apply any user overrides
    if overrides:
        config = config.update_from_dict(overrides)
        
    return config


def create_custom_config(base_config: str = "fast", **overrides) -> MFGRLConfig:
    """
    Create custom configuration starting from a base template.
    
    Args:
        base_config: Base configuration ("fast", "accurate", "research")
        **overrides: Configuration overrides
        
    Returns:
        Custom MFG-RL configuration
    """
    # Get base configuration
    if base_config == "fast":
        config = create_fast_config()
    elif base_config == "accurate":
        config = create_accurate_config()
    elif base_config == "research":
        config = create_research_config()
    else:
        raise ValueError(f"Unknown base config: {base_config}")
    
    # Apply overrides
    if overrides:
        config = config.update_from_dict(overrides)
    
    return config


def create_problem_specific_config(problem_type: str, **overrides) -> MFGRLConfig:
    """
    Create configuration optimized for specific problem types.
    
    Args:
        problem_type: Type of MFG problem ("linear_quadratic", "crowd_motion", etc.)
        **overrides: Configuration overrides
        
    Returns:
        Problem-specific MFG-RL configuration
    """
    if problem_type == "linear_quadratic":
        # Linear quadratic problems can use simpler networks
        base_overrides = {
            "network": {
                "hidden_layers": [32, 32],
                "activation": "tanh"  # Good for continuous control
            },
            "algorithm": {
                "learning_rate": 5e-4,
                "epsilon_decay": 0.99  # Faster exploration decay
            }
        }
    elif problem_type == "crowd_motion":
        # Crowd motion needs larger networks for complex interactions
        base_overrides = {
            "network": {
                "hidden_layers": [128, 64, 32],
                "activation": "relu",
                "dropout": 0.2
            },
            "algorithm": {
                "learning_rate": 1e-3,
                "population_sample_size": 3000  # Large population
            }
        }
    else:
        base_overrides = {}
    
    # Combine base overrides with user overrides
    combined_overrides = {**base_overrides, **overrides}
    
    return create_custom_config("fast", **combined_overrides)
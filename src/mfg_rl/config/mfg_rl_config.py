"""
Pydantic-based configuration system for MFG-RL.

This module provides type-safe configuration management following
the professional patterns from MFG_PDE.
"""

from typing import List, Optional, Union, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field, validator
from enum import Enum


class ActivationFunction(str, Enum):
    """Neural network activation functions."""
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    GELU = "gelu"
    SWISH = "swish"


class OptimizerType(str, Enum):
    """Optimizer types."""
    ADAM = "adam"
    SGD = "sgd"
    RMS_PROP = "rmsprop"
    ADAMW = "adamw"


class SchedulerType(str, Enum):
    """Learning rate schedulers."""
    CONSTANT = "constant"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    STEP = "step"
    PLATEAU = "plateau"


class NetworkConfig(BaseModel):
    """Neural network configuration."""
    hidden_layers: List[int] = Field(default=[64, 64], description="Hidden layer dimensions")
    activation: ActivationFunction = Field(default=ActivationFunction.RELU, description="Activation function")
    dropout: float = Field(default=0.0, ge=0.0, le=1.0, description="Dropout probability")
    batch_norm: bool = Field(default=False, description="Use batch normalization")
    layer_norm: bool = Field(default=False, description="Use layer normalization")
    initialization: str = Field(default="xavier_uniform", description="Weight initialization scheme")
    
    @validator('hidden_layers')
    def validate_hidden_layers(cls, v):
        if not v or any(dim <= 0 for dim in v):
            raise ValueError("All hidden layer dimensions must be positive")
        return v


class AlgorithmConfig(BaseModel):
    """Algorithm-specific configuration."""
    learning_rate: float = Field(default=1e-3, gt=0.0, description="Learning rate")
    batch_size: int = Field(default=32, gt=0, description="Batch size")
    memory_size: int = Field(default=10000, gt=0, description="Replay buffer size")
    target_update_frequency: int = Field(default=100, gt=0, description="Target network update frequency")
    
    # Exploration parameters
    epsilon_start: float = Field(default=1.0, ge=0.0, le=1.0, description="Initial exploration rate")
    epsilon_end: float = Field(default=0.01, ge=0.0, le=1.0, description="Final exploration rate")
    epsilon_decay: float = Field(default=0.995, ge=0.0, le=1.0, description="Exploration decay rate")
    
    # Optimization
    optimizer: OptimizerType = Field(default=OptimizerType.ADAM, description="Optimizer type")
    scheduler: SchedulerType = Field(default=SchedulerType.CONSTANT, description="Learning rate scheduler")
    grad_clip_norm: Optional[float] = Field(default=None, ge=0.0, description="Gradient clipping norm")
    
    # Algorithm-specific parameters
    discount_factor: float = Field(default=0.99, ge=0.0, le=1.0, description="Discount factor")
    tau: float = Field(default=0.005, gt=0.0, le=1.0, description="Soft update parameter")
    
    # Mean field specific
    mean_field_update_frequency: int = Field(default=10, gt=0, description="Mean field update frequency")
    population_sample_size: int = Field(default=1000, gt=0, description="Population sample size for mean field")
    
    @validator('epsilon_end')
    def epsilon_end_less_than_start(cls, v, values):
        if 'epsilon_start' in values and v > values['epsilon_start']:
            raise ValueError("epsilon_end must be <= epsilon_start")
        return v


class TrainingConfig(BaseModel):
    """Training configuration."""
    max_episodes: int = Field(default=1000, gt=0, description="Maximum training episodes")
    max_steps_per_episode: int = Field(default=200, gt=0, description="Maximum steps per episode")
    eval_frequency: int = Field(default=100, gt=0, description="Evaluation frequency")
    save_frequency: int = Field(default=500, gt=0, description="Model save frequency")
    
    # Convergence criteria
    convergence_threshold: float = Field(default=1e-4, gt=0.0, description="Convergence threshold")
    convergence_window: int = Field(default=100, gt=0, description="Window for convergence check")
    early_stopping_patience: int = Field(default=50, gt=0, description="Early stopping patience")
    
    # Parallel training
    num_workers: int = Field(default=1, ge=1, description="Number of parallel workers")
    use_multiprocessing: bool = Field(default=False, description="Use multiprocessing")


class LoggingConfig(BaseModel):
    """Logging and visualization configuration."""
    log_dir: Path = Field(default=Path("results/logs"), description="Logging directory")
    experiment_name: Optional[str] = Field(default=None, description="Experiment name")
    
    # Logging levels
    console_level: str = Field(default="INFO", description="Console logging level")
    file_level: str = Field(default="DEBUG", description="File logging level")
    
    # Tensorboard
    use_tensorboard: bool = Field(default=True, description="Use TensorBoard logging")
    tensorboard_dir: Path = Field(default=Path("results/tensorboard"), description="TensorBoard directory")
    
    # Weights & Biases
    use_wandb: bool = Field(default=False, description="Use Weights & Biases")
    wandb_project: Optional[str] = Field(default=None, description="W&B project name")
    wandb_entity: Optional[str] = Field(default=None, description="W&B entity")
    
    # Metrics logging
    log_frequency: int = Field(default=10, gt=0, description="Metrics logging frequency")
    save_trajectories: bool = Field(default=False, description="Save agent trajectories")
    save_mean_field: bool = Field(default=True, description="Save mean field evolution")
    
    # Plotting
    plot_frequency: int = Field(default=100, gt=0, description="Plotting frequency")
    save_plots: bool = Field(default=True, description="Save plots to disk")
    plot_format: str = Field(default="png", description="Plot file format")


class MFGRLConfig(BaseModel):
    """
    Main configuration class for MFG-RL.
    
    This class combines all configuration aspects and provides
    validation and serialization capabilities.
    """
    
    # Configuration sections
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Global settings
    seed: int = Field(default=42, ge=0, description="Random seed")
    device: str = Field(default="auto", description="Device (cpu/cuda/auto)")
    precision: str = Field(default="float32", description="Numerical precision")
    
    # Problem-specific overrides
    problem_overrides: Dict[str, Any] = Field(default_factory=dict, description="Problem-specific parameter overrides")
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"
        use_enum_values = True
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        import yaml
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'MFGRLConfig':
        """Load configuration from YAML file."""
        import yaml
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return cls(**config_dict)
    
    def update_from_dict(self, update_dict: Dict[str, Any]) -> 'MFGRLConfig':
        """Update configuration from dictionary."""
        config_dict = self.dict()
        
        def deep_update(d1, d2):
            for k, v in d2.items():
                if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                    deep_update(d1[k], v)
                else:
                    d1[k] = v
        
        deep_update(config_dict, update_dict)
        return self.__class__(**config_dict)
    
    def get_optimizer_kwargs(self) -> Dict[str, Any]:
        """Get optimizer keyword arguments."""
        base_kwargs = {"lr": self.algorithm.learning_rate}
        
        if self.algorithm.optimizer == OptimizerType.SGD:
            base_kwargs["momentum"] = 0.9
        elif self.algorithm.optimizer == OptimizerType.ADAMW:
            base_kwargs["weight_decay"] = 1e-4
            
        return base_kwargs
    
    def get_scheduler_kwargs(self) -> Dict[str, Any]:
        """Get scheduler keyword arguments."""
        if self.algorithm.scheduler == SchedulerType.EXPONENTIAL:
            return {"gamma": 0.95}
        elif self.algorithm.scheduler == SchedulerType.STEP:
            return {"step_size": 100, "gamma": 0.1}
        elif self.algorithm.scheduler == SchedulerType.COSINE:
            return {"T_max": self.training.max_episodes}
        else:
            return {}
    
    def validate_compatibility(self) -> None:
        """Validate configuration compatibility."""
        # Check device compatibility
        if self.device != "auto":
            import torch
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise ValueError("CUDA device specified but CUDA is not available")
        
        # Check memory constraints
        estimated_memory = (
            self.algorithm.batch_size * 
            sum(self.network.hidden_layers) * 4 / 1e6  # Rough estimate in MB
        )
        if estimated_memory > 1000:  # 1GB threshold
            import warnings
            warnings.warn(f"Estimated memory usage: {estimated_memory:.1f} MB may be too high")
    
    def __str__(self) -> str:
        """String representation for logging."""
        return f"""MFG-RL Configuration:
  Algorithm: {self.algorithm.optimizer.value} (lr={self.algorithm.learning_rate})
  Network: {self.network.hidden_layers} layers with {self.network.activation.value}
  Training: {self.training.max_episodes} episodes, batch_size={self.algorithm.batch_size}
  Device: {self.device}, Seed: {self.seed}"""
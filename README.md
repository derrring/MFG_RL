# MFG_RL: Reinforcement Learning for Mean Field Games

A comprehensive Python library for solving Mean Field Games (MFGs) using Reinforcement Learning approaches. This repository provides a unified framework for implementing and comparing different RL algorithms in the context of mean field games.

## 📋 Overview

Mean Field Games (MFGs) are a mathematical framework for modeling large-scale multi-agent systems where individual agents interact through a mean field (population distribution) rather than direct pairwise interactions. This library bridges the gap between the traditional PDE-based approaches and modern reinforcement learning techniques.

### Key Features

- 🚀 **Multiple RL Algorithms**: Implementations of DQN, Policy Gradient, and Actor-Critic methods adapted for MFGs
- 🌍 **Diverse Environments**: Linear-quadratic games, crowd motion models, and extensible base classes
- 🧠 **Neural Network Architectures**: Specialized networks for handling mean field interactions
- 📊 **Comprehensive Logging**: Integration with TensorBoard and Weights & Biases
- ⚡ **High Performance**: Optimized implementations using PyTorch/JAX
- 📚 **Educational**: Well-documented examples and tutorials

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9+ or TensorFlow 2.6+

### Install from source

```bash
git clone https://github.com/derrring/MFG_RL.git
cd MFG_RL
pip install -e .
```

### Development installation

```bash
git clone https://github.com/derrring/MFG_RL.git
cd MFG_RL
pip install -e ".[dev]"
```

## 🚀 Quick Start

Here's a simple example to get you started:

```python
from mfg_rl.environments import LinearQuadraticMFG
from mfg_rl.algorithms import MeanFieldDQN
from mfg_rl.utils import Config

# Configure the experiment
config = Config({
    'state_dim': 2,
    'action_dim': 1,
    'population_size': 100,
    'n_episodes': 1000
})

# Initialize environment and algorithm
env = LinearQuadraticMFG(config)
algorithm = MeanFieldDQN(config, env)

# Train the model
for episode in range(config.n_episodes):
    reward = algorithm.train_episode()
    if episode % 100 == 0:
        print(f"Episode {episode}, Reward: {reward:.3f}")
```

## 📁 Repository Structure

```
MFG_RL/
├── src/mfg_rl/              # Main package
│   ├── agents/              # RL agent implementations
│   ├── algorithms/          # MFG-specific RL algorithms
│   ├── environments/        # Mean field game environments
│   ├── networks/           # Neural network architectures
│   └── utils/              # Utilities and helper functions
├── examples/               # Example scripts and tutorials
│   ├── basic/             # Basic usage examples
│   └── advanced/          # Advanced implementations
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                # Test suite
├── config/               # Configuration files
├── data/                 # Data storage
│   ├── raw/             # Raw experimental data
│   └── processed/       # Processed results
├── results/             # Training outputs
│   ├── models/          # Saved model checkpoints
│   ├── logs/           # Training logs
│   └── plots/          # Visualization outputs
└── docs/               # Documentation
```

## 🤖 Supported Algorithms

### Deep Q-Learning (DQN) for MFGs
- **Mean Field DQN**: Adapts DQN to handle mean field interactions
- **Double DQN**: Reduces overestimation bias in MFG settings
- **Dueling DQN**: Separates state value and advantage estimation

### Policy Gradient Methods
- **Mean Field Policy Gradient**: Direct policy optimization with mean field coupling
- **Actor-Critic**: Combined value function approximation and policy optimization
- **Proximal Policy Optimization (PPO)**: Stable policy updates for MFGs

### Advanced Methods
- **Multi-Agent Actor-Critic**: Centralized training, decentralized execution
- **Mean Field Multi-Agent RL**: Population-based learning approaches

## 🎮 Environments

### Linear Quadratic MFG
Classic benchmark problem with quadratic costs and linear dynamics.

```python
from mfg_rl.environments import LinearQuadraticMFG
env = LinearQuadraticMFG(config)
```

### Crowd Motion
Pedestrian dynamics and crowd flow simulation.

```python
from mfg_rl.environments import CrowdMotion
env = CrowdMotion(config)
```

### Custom Environments
Extend the `BaseEnvironment` class to create your own MFG environments.

## 📖 Theoretical Background

Mean Field Games provide a mathematical framework for:

1. **Large Population Games**: Modeling systems with many interacting agents
2. **Nash Equilibrium**: Finding optimal strategies in competitive settings  
3. **Mean Field Coupling**: Agents interact through population statistics
4. **PDE Connections**: Links to Hamilton-Jacobi-Bellman and Fokker-Planck equations

The RL approach approximates the traditional PDE-based solution methods using:
- Function approximation via neural networks
- Sampling-based learning instead of solving PDEs directly
- Scalable algorithms for high-dimensional state/action spaces

## ⚖️ Comparison with PDE Methods

| Aspect | RL Approach (MFG_RL) | PDE Approach (MFG_PDE) |
|--------|---------------------|------------------------|
| **Scalability** | High-dimensional spaces | Limited by curse of dimensionality |
| **Flexibility** | Easy to modify rewards/dynamics | Requires analytical derivatives |
| **Convergence** | Approximate, empirical | Theoretical guarantees |
| **Implementation** | Straightforward | Complex numerical schemes |
| **Interpretability** | Limited | High theoretical insight |

## 🔬 Running Experiments

### Basic Linear Quadratic Example
```bash
python examples/basic/linear_quadratic_example.py
```

### Advanced Multi-Agent Scenarios
```bash
python examples/advanced/crowd_motion_experiment.py --config config/crowd_motion.yaml
```

### Custom Configuration
```bash
python examples/basic/linear_quadratic_example.py --config custom_config.yaml
```

## 📈 Monitoring and Visualization

### TensorBoard Integration
```bash
tensorboard --logdir results/logs
```

### Weights & Biases
Set `wandb: true` in your configuration file to enable automatic experiment tracking.

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/mfg_rl --cov-report=html

# Run specific test categories
pytest tests/test_algorithms.py
```

## 📖 Documentation

### API Documentation
```bash
cd docs/
make html
```

### Jupyter Notebooks
Explore the `notebooks/` directory for interactive tutorials and analysis examples.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and ensure they pass
5. Submit a pull request

### Code Style
We use Black for code formatting and flake8 for linting:

```bash
black src/ tests/
flake8 src/ tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📝 Citation

If you use this library in your research, please cite:

```bibtex
@software{mfg_rl2024,
  title={MFG_RL: Reinforcement Learning for Mean Field Games},
  author={derrring},
  year={2024},
  url={https://github.com/derrring/MFG_RL}
}
```

## 🔗 Related Work

- **MFG_PDE**: PDE-based approaches for Mean Field Games
- **Multi-Agent RL**: Traditional multi-agent reinforcement learning
- **Game Theory**: Classical game-theoretic solution concepts

## 🙏 Acknowledgments

- Mean Field Game theory pioneers: Lasry-Lions and Huang-Caines-Malhamé
- Open source RL community (Stable Baselines3, Ray RLlib)
- PyTorch and JAX development teams

---

**Note**: This is an active research project. APIs may change as the library evolves.
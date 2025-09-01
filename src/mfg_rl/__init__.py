"""
MFG_RL: Reinforcement Learning for Mean Field Games

A Python package for solving Mean Field Games using Reinforcement Learning approaches.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from . import agents
from . import algorithms  
from . import environments
from . import networks
from . import utils
from . import scoring
from . import mean_field
from . import policy

__all__ = [
    "agents",
    "algorithms", 
    "environments", 
    "networks",
    "utils",
    "scoring",
    "mean_field",
    "policy"
]
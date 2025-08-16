"""
Q-Optim: Hybrid Quantum-Classical Optimization Framework

A production-ready framework combining quantum algorithms with classical AI techniques
for solving complex combinatorial optimization problems.
"""

__version__ = "1.0.0"
__author__ = "Q-Optim Team"
__email__ = "info@qoptim.ai"

from .core.framework import QOptimFramework
from .core.problem import OptimizationProblem, ProblemType, Variable
from .core.solution import Solution
from .quantum.qaoa import QAOA
from .quantum.vqe import VQE
from .quantum.grover import GroverSearch
from .ai.preprocessing import AIPreprocessor
from .classical.solvers import ClassicalSolvers

__all__ = [
    "QOptimFramework",
    "OptimizationProblem", 
    "ProblemType",
    "Variable",
    "Solution",
    "QAOA",
    "VQE",
    "GroverSearch",
    "AIPreprocessor",
    "ClassicalSolvers",
]

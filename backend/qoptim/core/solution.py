"""
Solution representation and comparison utilities for optimization problems.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json


class SolutionStatus(Enum):
    """Status of an optimization solution."""
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    ERROR = "error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class SolverType(Enum):
    """Type of solver used to find the solution."""
    QAOA = "qaoa"
    VQE = "vqe"
    GROVER = "grover"
    SIMULATED_ANNEALING = "simulated_annealing"
    GENETIC_ALGORITHM = "genetic_algorithm"
    EXACT = "exact"
    HYBRID = "hybrid"
    CLASSICAL = "classical"
    QUANTUM = "quantum"


@dataclass
class SolutionMetadata:
    """Metadata associated with a solution."""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    execution_time: float = 0.0
    iterations: int = 0
    function_evaluations: int = 0
    quantum_shots: int = 0
    convergence_tolerance: float = 0.0
    memory_usage: float = 0.0
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'execution_time': self.execution_time,
            'iterations': self.iterations,
            'function_evaluations': self.function_evaluations,
            'quantum_shots': self.quantum_shots,
            'convergence_tolerance': self.convergence_tolerance,
            'memory_usage': self.memory_usage,
            'additional_info': self.additional_info
        }


class Solution:
    """
    Comprehensive solution representation for optimization problems.
    
    Contains solution variables, objective value, metadata, and comparison utilities.
    """
    
    def __init__(self, problem_name: str, solver_type: SolverType, status: SolutionStatus = SolutionStatus.UNKNOWN):
        self.problem_name = problem_name
        self.solver_type = solver_type
        self.status = status
        
        # Solution data
        self.variables: Dict[str, float] = {}
        self.binary_solution: Optional[np.ndarray] = None
        self.objective_value: Optional[float] = None
        
        # Quality metrics
        self.feasibility_score: float = 0.0
        self.confidence_score: float = 0.0
        self.quantum_advantage: bool = False
        
        # Constraint information
        self.constraint_violations: List[Dict[str, Any]] = []
        self.penalty_value: float = 0.0
        
        # Benchmarking
        self.classical_baseline: Optional[float] = None
        self.benchmarks: Dict[str, Dict[str, Any]] = {}
        
        # Alternative solutions
        self.alternative_solutions: List['Solution'] = []
        
        # Metadata
        self.metadata = SolutionMetadata()
        
        # Solution history (for iterative algorithms)
        self.solution_history: List[Dict[str, Any]] = []
    
    def set_variable_values(self, variables: Dict[str, float]) -> None:
        """Set solution variable values."""
        self.variables = variables.copy()
    
    def set_binary_solution(self, binary_solution: np.ndarray) -> None:
        """Set binary solution representation."""
        self.binary_solution = binary_solution.copy()
    
    def set_objective_value(self, objective_value: float) -> None:
        """Set objective function value."""
        self.objective_value = objective_value
    
    def set_classical_baseline(self, baseline_value: float) -> None:
        """Set classical baseline for quantum advantage calculation."""
        self.classical_baseline = baseline_value
        self._update_quantum_advantage()
    
    def add_constraint_violation(
        self, 
        constraint_name: str, 
        violation_amount: float, 
        constraint_type: str
    ) -> None:
        """Add constraint violation information."""
        violation = {
            'constraint_name': constraint_name,
            'violation_amount': violation_amount,
            'constraint_type': constraint_type
        }
        self.constraint_violations.append(violation)
        self.penalty_value += violation_amount
    
    def add_benchmark(self, method_name: str, benchmark_data: Dict[str, Any]) -> None:
        """Add benchmark comparison data."""
        self.benchmarks[method_name] = benchmark_data
    
    def add_alternative_solution(self, solution: 'Solution') -> None:
        """Add alternative solution for comparison."""
        self.alternative_solutions.append(solution)
    
    def is_feasible(self) -> bool:
        """Check if solution is feasible (no constraint violations)."""
        return len(self.constraint_violations) == 0 and self.status in [
            SolutionStatus.OPTIMAL, SolutionStatus.FEASIBLE
        ]
    
    def calculate_feasibility_score(self) -> float:
        """Calculate feasibility score based on constraint violations."""
        if not self.constraint_violations:
            self.feasibility_score = 1.0
        else:
            # Simple scoring: 1 - normalized penalty
            max_penalty = sum(v['violation_amount'] for v in self.constraint_violations)
            self.feasibility_score = max(0.0, 1.0 - max_penalty / (max_penalty + 1.0))
        
        return self.feasibility_score
    
    def finalize_timing(self) -> None:
        """Finalize timing calculations."""
        if self.metadata.start_time and self.metadata.end_time:
            self.metadata.execution_time = self.metadata.end_time - self.metadata.start_time
        elif self.metadata.start_time:
            self.metadata.execution_time = time.time() - self.metadata.start_time
    
    def update_metadata(self, **kwargs) -> None:
        """Update solution metadata."""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
            else:
                self.metadata.additional_info[key] = value
    
    def _update_quantum_advantage(self) -> None:
        """Update quantum advantage flag based on performance comparison."""
        if (self.classical_baseline is not None and 
            self.objective_value is not None and 
            self.solver_type in [SolverType.QAOA, SolverType.VQE, SolverType.GROVER]):
            
            # For minimization problems (assuming most are minimization)
            self.quantum_advantage = self.objective_value < self.classical_baseline
    
    def get_solution_quality(self) -> float:
        """Get overall solution quality score (0-1)."""
        quality_components = []
        
        # Feasibility component
        quality_components.append(self.feasibility_score)
        
        # Confidence component
        quality_components.append(self.confidence_score)
        
        # Status component
        status_scores = {
            SolutionStatus.OPTIMAL: 1.0,
            SolutionStatus.FEASIBLE: 0.8,
            SolutionStatus.INFEASIBLE: 0.0,
            SolutionStatus.ERROR: 0.0,
            SolutionStatus.TIMEOUT: 0.3,
            SolutionStatus.UNKNOWN: 0.1
        }
        quality_components.append(status_scores.get(self.status, 0.1))
        
        return np.mean(quality_components)
    
    def add_solution_history_entry(self, iteration: int, variables: Dict[str, float], objective_value: float) -> None:
        """Add entry to solution history for iterative algorithms."""
        entry = {
            'iteration': iteration,
            'variables': variables.copy(),
            'objective_value': objective_value,
            'timestamp': time.time()
        }
        self.solution_history.append(entry)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert solution to dictionary representation."""
        return {
            'problem_name': self.problem_name,
            'solver_type': self.solver_type.value,
            'status': self.status.value,
            'variables': self.variables,
            'binary_solution': self.binary_solution.tolist() if self.binary_solution is not None else None,
            'objective_value': self.objective_value,
            'feasibility_score': self.feasibility_score,
            'confidence_score': self.confidence_score,
            'quantum_advantage': self.quantum_advantage,
            'constraint_violations': self.constraint_violations,
            'penalty_value': self.penalty_value,
            'classical_baseline': self.classical_baseline,
            'benchmarks': self.benchmarks,
            'metadata': self.metadata.to_dict(),
            'solution_quality': self.get_solution_quality()
        }
    
    def to_json(self) -> str:
        """Convert solution to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Solution':
        """Create solution from dictionary."""
        solution = cls(
            problem_name=data['problem_name'],
            solver_type=SolverType(data['solver_type']),
            status=SolutionStatus(data['status'])
        )
        
        solution.variables = data.get('variables', {})
        
        if data.get('binary_solution'):
            solution.binary_solution = np.array(data['binary_solution'])
        
        solution.objective_value = data.get('objective_value')
        solution.feasibility_score = data.get('feasibility_score', 0.0)
        solution.confidence_score = data.get('confidence_score', 0.0)
        solution.quantum_advantage = data.get('quantum_advantage', False)
        solution.constraint_violations = data.get('constraint_violations', [])
        solution.penalty_value = data.get('penalty_value', 0.0)
        solution.classical_baseline = data.get('classical_baseline')
        solution.benchmarks = data.get('benchmarks', {})
        
        # Restore metadata
        metadata_dict = data.get('metadata', {})
        for key, value in metadata_dict.items():
            if hasattr(solution.metadata, key):
                setattr(solution.metadata, key, value)
        
        return solution
    
    def __str__(self) -> str:
        """String representation of solution."""
        return (
            f"Solution(problem='{self.problem_name}', "
            f"solver={self.solver_type.value}, status={self.status.value}, "
            f"objective={self.objective_value}, feasible={self.is_feasible()})"
        )
    
    def __repr__(self) -> str:
        """Detailed representation of solution."""
        return self.__str__()


class SolutionComparator:
    """Utility class for comparing and ranking solutions."""
    
    @staticmethod
    def compare_solutions(solution1: Solution, solution2: Solution, minimize: bool = True) -> int:
        """
        Compare two solutions.
        
        Returns:
            -1 if solution1 is better than solution2
             0 if solutions are equivalent
             1 if solution2 is better than solution1
        """
        # First check feasibility
        feas1, feas2 = solution1.is_feasible(), solution2.is_feasible()
        
        if feas1 and not feas2:
            return -1
        elif feas2 and not feas1:
            return 1
        elif not feas1 and not feas2:
            # Both infeasible, compare by constraint violation
            if solution1.penalty_value < solution2.penalty_value:
                return -1
            elif solution1.penalty_value > solution2.penalty_value:
                return 1
            else:
                return 0
        
        # Both feasible, compare by objective value
        obj1, obj2 = solution1.objective_value, solution2.objective_value
        
        if obj1 is None and obj2 is None:
            return 0
        elif obj1 is None:
            return 1
        elif obj2 is None:
            return -1
        
        if minimize:
            if obj1 < obj2:
                return -1
            elif obj1 > obj2:
                return 1
        else:
            if obj1 > obj2:
                return -1
            elif obj1 < obj2:
                return 1
        
        return 0
    
    @staticmethod
    def rank_solutions(solutions: List[Solution], minimize: bool = True) -> List[Solution]:
        """
        Rank solutions from best to worst.
        
        Args:
            solutions: List of solutions to rank
            minimize: Whether to minimize or maximize objective
            
        Returns:
            Sorted list of solutions (best to worst)
        """
        import functools
        
        comparison_func = functools.partial(
            SolutionComparator.compare_solutions, 
            minimize=minimize
        )
        
        return sorted(solutions, key=functools.cmp_to_key(comparison_func))
    
    @staticmethod
    def get_pareto_front(solutions: List[Solution]) -> List[Solution]:
        """
        Find Pareto-optimal solutions based on multiple criteria.
        
        Criteria: objective value, feasibility score, confidence score
        """
        if not solutions:
            return []
        
        pareto_solutions = []
        
        for candidate in solutions:
            is_dominated = False
            
            for other in solutions:
                if candidate == other:
                    continue
                
                # Check if other solution dominates candidate
                if SolutionComparator._dominates(other, candidate):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_solutions.append(candidate)
        
        return pareto_solutions
    
    @staticmethod
    def _dominates(solution1: Solution, solution2: Solution) -> bool:
        """Check if solution1 dominates solution2 in Pareto sense."""
        # Get criteria values
        criteria1 = SolutionComparator._get_criteria_values(solution1)
        criteria2 = SolutionComparator._get_criteria_values(solution2)
        
        # Check if solution1 is at least as good in all criteria
        # and strictly better in at least one
        at_least_as_good = all(c1 >= c2 for c1, c2 in zip(criteria1, criteria2))
        strictly_better = any(c1 > c2 for c1, c2 in zip(criteria1, criteria2))
        
        return at_least_as_good and strictly_better
    
    @staticmethod
    def _get_criteria_values(solution: Solution) -> Tuple[float, float, float]:
        """Get criteria values for Pareto comparison."""
        # Objective value (negative for minimization problems)
        obj_val = -solution.objective_value if solution.objective_value is not None else -float('inf')
        
        return (
            obj_val,
            solution.feasibility_score,
            solution.confidence_score
        )


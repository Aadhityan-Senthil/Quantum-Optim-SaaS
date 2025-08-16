"""
Optimization problem definition and data structures.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx


class ProblemType(Enum):
    """Types of optimization problems supported by Q-Optim."""
    COMBINATORIAL = "combinatorial"
    CONTINUOUS = "continuous"
    MIXED_INTEGER = "mixed_integer"
    SCHEDULING = "scheduling"
    PORTFOLIO = "portfolio"
    TSP = "traveling_salesman"
    GRAPH_COLORING = "graph_coloring"
    MAX_CUT = "max_cut"
    KNAPSACK = "knapsack"
    SUPPLY_CHAIN = "supply_chain"


class OptimizationType(Enum):
    """Optimization objectives."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class Constraint:
    """A single constraint in the optimization problem."""
    name: str
    type: str  # "equality", "inequality", "bounds"
    expression: Any  # Mathematical expression or function
    value: float
    variables: List[str] = field(default_factory=list)
    
    def evaluate(self, solution: Dict[str, float]) -> bool:
        """Evaluate if the constraint is satisfied by a solution."""
        if callable(self.expression):
            return self.expression(solution)
        # Default implementation for simple constraints
        return True


@dataclass
class Variable:
    """A decision variable in the optimization problem."""
    name: str
    type: str  # "binary", "integer", "continuous"
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    initial_value: Optional[float] = None


class OptimizationProblem:
    """
    Core optimization problem representation that supports various problem types
    and can be converted to QUBO/Ising formats for quantum solving.
    """
    
    def __init__(
        self,
        name: str,
        problem_type: ProblemType,
        optimization_type: OptimizationType = OptimizationType.MINIMIZE
    ):
        self.name = name
        self.problem_type = problem_type
        self.optimization_type = optimization_type
        
        # Problem components
        self.variables: Dict[str, Variable] = {}
        self.constraints: List[Constraint] = []
        self.objective_function: Optional[Any] = None
        
        # Problem data
        self.data: Dict[str, Any] = {}
        self.graph: Optional[nx.Graph] = None
        
        # QUBO/Ising representation
        self._qubo_matrix: Optional[np.ndarray] = None
        self._ising_h: Optional[np.ndarray] = None
        self._ising_j: Optional[np.ndarray] = None
        
        # Metadata
        self.metadata: Dict[str, Any] = {}
        self.preprocessing_info: Dict[str, Any] = {}
    
    def add_variable(self, variable: Variable) -> None:
        """Add a decision variable to the problem."""
        self.variables[variable.name] = variable
    
    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the problem."""
        self.constraints.append(constraint)
    
    def set_objective(self, objective_function: Any) -> None:
        """Set the objective function."""
        self.objective_function = objective_function
    
    def set_data(self, key: str, value: Any) -> None:
        """Set problem-specific data."""
        self.data[key] = value
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """Get problem-specific data."""
        return self.data.get(key, default)
    
    def set_graph(self, graph: nx.Graph) -> None:
        """Set graph representation for graph-based problems."""
        self.graph = graph
    
    def to_qubo(self) -> Tuple[np.ndarray, float]:
        """
        Convert the problem to QUBO (Quadratic Unconstrained Binary Optimization) format.
        Returns Q matrix and offset.
        """
        if self._qubo_matrix is not None:
            return self._qubo_matrix, self.metadata.get('qubo_offset', 0.0)
        
        # Problem-specific QUBO conversion
        if self.problem_type == ProblemType.MAX_CUT:
            return self._maxcut_to_qubo()
        elif self.problem_type == ProblemType.TSP:
            return self._tsp_to_qubo()
        elif self.problem_type == ProblemType.KNAPSACK:
            return self._knapsack_to_qubo()
        else:
            # Generic conversion based on objective and constraints
            return self._generic_to_qubo()
    
    def to_ising(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Convert the problem to Ising model format.
        Returns h (linear terms), J (quadratic terms), and offset.
        """
        if self._ising_h is not None and self._ising_j is not None:
            return self._ising_h, self._ising_j, self.metadata.get('ising_offset', 0.0)
        
        # Convert QUBO to Ising
        Q, offset = self.to_qubo()
        h, J, ising_offset = self._qubo_to_ising(Q, offset)
        
        self._ising_h = h
        self._ising_j = J
        self.metadata['ising_offset'] = ising_offset
        
        return h, J, ising_offset
    
    def _maxcut_to_qubo(self) -> Tuple[np.ndarray, float]:
        """Convert Max-Cut problem to QUBO format."""
        if self.graph is None:
            raise ValueError("Graph must be set for Max-Cut problem")
        
        n = self.graph.number_of_nodes()
        Q = np.zeros((n, n))
        
        for i, j, data in self.graph.edges(data=True):
            weight = data.get('weight', 1.0)
            Q[i, i] -= weight
            Q[j, j] -= weight
            Q[i, j] += 2 * weight
        
        self._qubo_matrix = Q
        return Q, 0.0
    
    def _tsp_to_qubo(self) -> Tuple[np.ndarray, float]:
        """Convert Traveling Salesman Problem to QUBO format."""
        distance_matrix = self.get_data('distance_matrix')
        if distance_matrix is None:
            raise ValueError("Distance matrix required for TSP")
        
        n = len(distance_matrix)
        penalty = self.get_data('penalty_weight', 10.0)
        
        # Create QUBO matrix for TSP
        # Variables: x_ij = 1 if city i is visited at time j
        num_vars = n * n
        Q = np.zeros((num_vars, num_vars))
        
        # Objective: minimize travel distance
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != k:
                        var1 = i * n + j
                        var2 = k * n + ((j + 1) % n)
                        Q[var1, var2] += distance_matrix[i][k]
        
        # Constraints: each city visited exactly once
        for i in range(n):
            for j in range(n):
                for k in range(j + 1, n):
                    var1 = i * n + j
                    var2 = i * n + k
                    Q[var1, var2] += penalty
        
        # Constraints: each time slot has exactly one city
        for j in range(n):
            for i in range(n):
                for k in range(i + 1, n):
                    var1 = i * n + j
                    var2 = k * n + j
                    Q[var1, var2] += penalty
        
        self._qubo_matrix = Q
        return Q, 0.0
    
    def _knapsack_to_qubo(self) -> Tuple[np.ndarray, float]:
        """Convert Knapsack problem to QUBO format."""
        weights = self.get_data('weights')
        values = self.get_data('values') 
        capacity = self.get_data('capacity')
        
        if not all([weights, values, capacity]):
            raise ValueError("Weights, values, and capacity required for Knapsack")
        
        n = len(weights)
        penalty = self.get_data('penalty_weight', 10.0)
        
        Q = np.zeros((n, n))
        
        # Objective: maximize value (convert to minimization)
        for i in range(n):
            Q[i, i] -= values[i]
        
        # Constraint: weight limit (using penalty method)
        for i in range(n):
            for j in range(n):
                Q[i, j] += penalty * weights[i] * weights[j]
        
        for i in range(n):
            Q[i, i] -= 2 * penalty * capacity * weights[i]
        
        self._qubo_matrix = Q
        return Q, penalty * capacity**2
    
    def _generic_to_qubo(self) -> Tuple[np.ndarray, float]:
        """Generic QUBO conversion for custom problems."""
        n = len(self.variables)
        Q = np.zeros((n, n))
        
        # This is a placeholder for custom QUBO generation
        # In practice, this would analyze the objective function and constraints
        
        if callable(self.objective_function):
            # Try to extract quadratic terms from objective function
            pass
        
        self._qubo_matrix = Q
        return Q, 0.0
    
    def _qubo_to_ising(self, Q: np.ndarray, offset: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Convert QUBO to Ising format using transformation x = (s + 1)/2."""
        n = Q.shape[0]
        
        # Linear terms h_i
        h = np.zeros(n)
        for i in range(n):
            h[i] = Q[i, i] + sum(Q[i, j] + Q[j, i] for j in range(n) if j != i) / 2
        h = h / 2
        
        # Quadratic terms J_ij  
        J = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                J[i, j] = (Q[i, j] + Q[j, i]) / 4
        
        # Offset
        ising_offset = offset + sum(Q[i, i] for i in range(n)) / 2 + sum(
            Q[i, j] + Q[j, i] for i in range(n) for j in range(i + 1, n)
        ) / 4
        
        return h, J, ising_offset
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the problem definition."""
        errors = []
        
        if not self.variables:
            errors.append("Problem must have at least one variable")
        
        if self.objective_function is None:
            errors.append("Objective function must be set")
        
        # Problem-specific validation
        if self.problem_type == ProblemType.MAX_CUT and self.graph is None:
            errors.append("Max-Cut problem requires a graph")
        
        if self.problem_type == ProblemType.TSP and 'distance_matrix' not in self.data:
            errors.append("TSP problem requires distance matrix")
        
        if self.problem_type == ProblemType.KNAPSACK:
            required_data = ['weights', 'values', 'capacity']
            for key in required_data:
                if key not in self.data:
                    errors.append(f"Knapsack problem requires {key}")
        
        return len(errors) == 0, errors
    
    def get_problem_size(self) -> Dict[str, int]:
        """Get problem size metrics."""
        return {
            'num_variables': len(self.variables),
            'num_constraints': len(self.constraints),
            'num_binary_vars': sum(1 for v in self.variables.values() if v.type == 'binary'),
            'num_continuous_vars': sum(1 for v in self.variables.values() if v.type == 'continuous'),
            'num_integer_vars': sum(1 for v in self.variables.values() if v.type == 'integer'),
        }
    
    def __str__(self) -> str:
        """String representation of the problem."""
        size_info = self.get_problem_size()
        return (
            f"OptimizationProblem(name='{self.name}', "
            f"type={self.problem_type.value}, "
            f"variables={size_info['num_variables']}, "
            f"constraints={size_info['num_constraints']})"
        )

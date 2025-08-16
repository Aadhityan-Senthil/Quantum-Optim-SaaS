"""
Grover's Search Algorithm implementation for unstructured search problems.
"""

import numpy as np
import time
import math
from typing import Dict, List, Optional, Any, Tuple, Union, Callable

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit_aer import AerSimulator
    import pennylane as qml
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from ..core.problem import OptimizationProblem, ProblemType
from ..core.solution import Solution, SolutionStatus, SolverType
from ..utils.config import Config
from ..utils.logging import get_logger


class GroverSearch:
    """
    Grover's Search Algorithm for finding solutions in unstructured search spaces.
    
    Particularly useful for problems where we need to find specific states
    that satisfy certain conditions (e.g., graph coloring, constraint satisfaction).
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Grover parameters
        self.shots = config.get('quantum_shots', 1024)
        self.backend_name = config.get('quantum_backend', 'qasm_simulator')
        self.use_pennylane = config.get('use_pennylane', False)
        
        # Initialize quantum backend
        if QISKIT_AVAILABLE:
            self._init_qiskit_backend()
        
        self.logger.info("Grover Search initialized")
    
    def _init_qiskit_backend(self):
        """Initialize Qiskit backend."""
        try:
            if self.backend_name == 'qasm_simulator':
                self.backend = AerSimulator()
            else:
                from qiskit_ibm_runtime import QiskitRuntimeService
                service = QiskitRuntimeService()
                self.backend = service.backend(self.backend_name)
            
            self.logger.info(f"Initialized Qiskit backend: {self.backend_name}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Qiskit backend: {e}")
            self.backend = None
    
    def solve(self, problem: OptimizationProblem) -> Solution:
        """
        Solve optimization problem using Grover's algorithm.
        
        Args:
            problem: The optimization problem to solve
            
        Returns:
            Solution object with Grover results
        """
        self.logger.info(f"Starting Grover search for problem: {problem.name}")
        start_time = time.time()
        
        solution = Solution(problem.name, SolverType.GROVER)
        solution.metadata.start_time = start_time
        
        try:
            # Determine search space size
            if problem.problem_type == ProblemType.GRAPH_COLORING:
                result = self._solve_graph_coloring(problem)
            elif problem.problem_type == ProblemType.MAX_CUT:
                result = self._solve_max_cut(problem)
            else:
                # Convert to general search problem
                result = self._solve_general_search(problem)
            
            # Process results
            if result is not None:
                self._process_results(solution, result, problem)
                solution.status = SolutionStatus.FEASIBLE
            else:
                solution.status = SolutionStatus.ERROR
                
        except Exception as e:
            self.logger.error(f"Grover search failed: {e}")
            solution.status = SolutionStatus.ERROR
            solution.update_metadata(error_message=str(e))
        
        finally:
            solution.metadata.end_time = time.time()
            solution.finalize_timing()
            
        self.logger.info(f"Grover search completed in {solution.metadata.execution_time:.4f}s")
        return solution
    
    def _solve_graph_coloring(self, problem: OptimizationProblem) -> Optional[Dict[str, Any]]:
        """Solve graph coloring using Grover's algorithm."""
        graph = problem.graph
        num_colors = problem.get_data('num_colors', 3)
        
        if graph is None:
            self.logger.error("Graph coloring requires a graph")
            return None
        
        n_nodes = graph.number_of_nodes()
        n_qubits = n_nodes * math.ceil(math.log2(num_colors))
        
        if n_qubits > self.config.get('max_qubits', 20):
            self.logger.error(f"Problem size {n_qubits} exceeds maximum qubits limit")
            return None
        
        # Define oracle for valid coloring
        def coloring_oracle(bitstring):
            # Convert bitstring to coloring
            coloring = self._bitstring_to_coloring(bitstring, n_nodes, num_colors)
            
            # Check if it's a valid coloring
            for u, v in graph.edges():
                if coloring[u] == coloring[v]:
                    return False
            return True
        
        if self.use_pennylane:
            return self._grover_search_pennylane(n_qubits, coloring_oracle)
        else:
            return self._grover_search_qiskit(n_qubits, coloring_oracle)
    
    def _solve_max_cut(self, problem: OptimizationProblem) -> Optional[Dict[str, Any]]:
        """Solve Max-Cut using Grover's algorithm to find good cuts."""
        graph = problem.graph
        target_cut_value = problem.get_data('target_cut_value')
        
        if graph is None:
            self.logger.error("Max-Cut requires a graph")
            return None
        
        n_nodes = graph.number_of_nodes()
        n_qubits = n_nodes
        
        if n_qubits > self.config.get('max_qubits', 20):
            self.logger.error(f"Problem size {n_qubits} exceeds maximum qubits limit")
            return None
        
        # Define oracle for good cuts
        def cut_oracle(bitstring):
            # Calculate cut value
            cut_value = 0
            for u, v, data in graph.edges(data=True):
                weight = data.get('weight', 1.0)
                if int(bitstring[u]) != int(bitstring[v]):
                    cut_value += weight
            
            # Return True if cut value is above threshold
            threshold = target_cut_value if target_cut_value else cut_value > len(graph.edges()) * 0.7
            return cut_value >= threshold
        
        if self.use_pennylane:
            return self._grover_search_pennylane(n_qubits, cut_oracle)
        else:
            return self._grover_search_qiskit(n_qubits, cut_oracle)
    
    def _solve_general_search(self, problem: OptimizationProblem) -> Optional[Dict[str, Any]]:
        """Solve general search problem using Grover's algorithm."""
        # For general problems, estimate search space from variables
        n_vars = len(problem.variables) if problem.variables else 4
        n_qubits = n_vars
        
        if n_qubits > self.config.get('max_qubits', 20):
            self.logger.error(f"Problem size {n_qubits} exceeds maximum qubits limit")
            return None
        
        # Generic oracle based on constraints
        def generic_oracle(bitstring):
            # Convert to variable values
            var_values = {f'x_{i}': int(bitstring[i]) for i in range(n_qubits)}
            
            # Check constraints
            for constraint in problem.constraints:
                try:
                    if not constraint.evaluate(var_values):
                        return False
                except:
                    pass
            
            return True
        
        if self.use_pennylane:
            return self._grover_search_pennylane(n_qubits, generic_oracle)
        else:
            return self._grover_search_qiskit(n_qubits, generic_oracle)
    
    def _grover_search_qiskit(self, n_qubits: int, oracle_func: Callable) -> Optional[Dict[str, Any]]:
        """Perform Grover search using Qiskit."""
        try:
            # Estimate number of solutions (assume small fraction)
            N = 2**n_qubits
            M = max(1, N // 8)  # Assume 1/8 of solutions are valid
            
            # Calculate optimal number of iterations
            optimal_iterations = math.floor(math.pi * math.sqrt(N/M) / 4)
            optimal_iterations = max(1, min(optimal_iterations, 10))  # Limit iterations
            
            # Create Grover circuit
            circuit = self._create_grover_circuit_qiskit(n_qubits, oracle_func, optimal_iterations)
            
            # Execute circuit
            from qiskit import execute
            job = execute(circuit, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            
            # Find most frequent result
            best_result = max(counts, key=counts.get)
            
            return {
                'best_solution': best_result,
                'measurement_counts': counts,
                'iterations': optimal_iterations,
                'search_space_size': N,
                'estimated_solutions': M
            }
            
        except Exception as e:
            self.logger.error(f"Qiskit Grover search failed: {e}")
            return None
    
    def _grover_search_pennylane(self, n_qubits: int, oracle_func: Callable) -> Optional[Dict[str, Any]]:
        """Perform Grover search using PennyLane."""
        try:
            # Estimate parameters
            N = 2**n_qubits
            M = max(1, N // 8)
            optimal_iterations = math.floor(math.pi * math.sqrt(N/M) / 4)
            optimal_iterations = max(1, min(optimal_iterations, 10))
            
            # Create device
            dev = qml.device('default.qubit', wires=n_qubits, shots=self.shots)
            
            @qml.qnode(dev)
            def grover_circuit():
                # Initialize superposition
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                
                # Grover iterations
                for _ in range(optimal_iterations):
                    # Oracle
                    self._apply_oracle_pennylane(oracle_func, n_qubits)
                    
                    # Diffusion operator
                    self._apply_diffusion_pennylane(n_qubits)
                
                # Measurement
                return [qml.sample(qml.PauliZ(i)) for i in range(n_qubits)]
            
            # Execute circuit multiple times
            samples = []
            for _ in range(self.shots):
                result = grover_circuit()
                bitstring = ''.join(['0' if x == 1 else '1' for x in result])
                samples.append(bitstring)
            
            # Count results
            from collections import Counter
            counts = Counter(samples)
            best_result = max(counts, key=counts.get)
            
            return {
                'best_solution': best_result,
                'measurement_counts': dict(counts),
                'iterations': optimal_iterations,
                'search_space_size': N,
                'estimated_solutions': M
            }
            
        except Exception as e:
            self.logger.error(f"PennyLane Grover search failed: {e}")
            return None
    
    def _create_grover_circuit_qiskit(
        self, 
        n_qubits: int, 
        oracle_func: Callable, 
        iterations: int
    ) -> QuantumCircuit:
        """Create Grover circuit using Qiskit."""
        qreg = QuantumRegister(n_qubits, 'q')
        creg = ClassicalRegister(n_qubits, 'c')
        circuit = QuantumCircuit(qreg, creg)
        
        # Initialize superposition
        circuit.h(qreg)
        
        # Grover iterations
        for _ in range(iterations):
            # Oracle (simplified - marks all states)
            # In practice, this would be problem-specific
            circuit.z(qreg[0])  # Placeholder oracle
            
            # Diffusion operator
            circuit.h(qreg)
            circuit.x(qreg)
            circuit.h(qreg[-1])
            circuit.mct(qreg[:-1], qreg[-1])  # Multi-controlled Toffoli
            circuit.h(qreg[-1])
            circuit.x(qreg)
            circuit.h(qreg)
        
        # Measurement
        circuit.measure(qreg, creg)
        
        return circuit
    
    def _apply_oracle_pennylane(self, oracle_func: Callable, n_qubits: int) -> None:
        """Apply oracle in PennyLane (simplified implementation)."""
        # This is a simplified oracle - in practice, oracles are problem-specific
        # and require more sophisticated implementation
        qml.PauliZ(wires=0)  # Placeholder
    
    def _apply_diffusion_pennylane(self, n_qubits: int) -> None:
        """Apply diffusion operator in PennyLane."""
        # Apply Hadamard
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        
        # Apply X gates
        for i in range(n_qubits):
            qml.PauliX(wires=i)
        
        # Multi-controlled Z
        if n_qubits > 1:
            control_wires = list(range(n_qubits - 1))
            target_wire = n_qubits - 1
            qml.MultiControlledX(wires=control_wires + [target_wire])
        
        # Undo X gates
        for i in range(n_qubits):
            qml.PauliX(wires=i)
        
        # Apply Hadamard
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
    
    def _bitstring_to_coloring(self, bitstring: str, n_nodes: int, num_colors: int) -> List[int]:
        """Convert bitstring to graph coloring."""
        bits_per_node = math.ceil(math.log2(num_colors))
        coloring = []
        
        for i in range(n_nodes):
            start_idx = i * bits_per_node
            end_idx = start_idx + bits_per_node
            
            if end_idx <= len(bitstring):
                node_bits = bitstring[start_idx:end_idx]
                color = int(node_bits, 2) % num_colors
                coloring.append(color)
            else:
                coloring.append(0)  # Default color
        
        return coloring
    
    def _process_results(
        self, 
        solution: Solution, 
        result: Dict[str, Any], 
        problem: OptimizationProblem
    ) -> None:
        """Process Grover results and update solution object."""
        best_solution = result['best_solution']
        
        # Convert bitstring to variable values
        variable_values = {}
        binary_solution = np.array([int(b) for b in best_solution])
        
        for i in range(len(best_solution)):
            variable_values[f'x_{i}'] = float(binary_solution[i])
        
        solution.set_variable_values(variable_values)
        solution.set_binary_solution(binary_solution)
        
        # Calculate objective value if possible
        if callable(problem.objective_function):
            try:
                obj_value = problem.objective_function(variable_values)
                solution.set_objective_value(obj_value)
            except:
                solution.set_objective_value(0.0)
        
        # Update metadata
        solution.update_metadata(
            iterations=result.get('iterations', 0),
            quantum_shots=self.shots,
            search_space_size=result.get('search_space_size', 0),
            estimated_solutions=result.get('estimated_solutions', 0),
            measurement_counts=result.get('measurement_counts', {})
        )
        
        # Set confidence based on measurement frequency
        counts = result.get('measurement_counts', {})
        total_shots = sum(counts.values()) if counts else 1
        best_count = counts.get(best_solution, 1)
        solution.confidence_score = best_count / total_shots
        
        self.logger.info(f"Grover solution processed: confidence={solution.confidence_score:.3f}")

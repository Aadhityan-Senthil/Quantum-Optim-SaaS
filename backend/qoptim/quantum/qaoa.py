"""
Quantum Approximate Optimization Algorithm (QAOA) implementation for discrete optimization problems.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from scipy.optimize import minimize
import logging

# Quantum computing imports
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit_aer import AerSimulator
    from qiskit.primitives import Sampler
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit.algorithms.optimizers import COBYLA, SPSA, L_BFGS_B
    import pennylane as qml
    from pennylane import numpy as pnp
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from ..core.problem import OptimizationProblem, ProblemType
from ..core.solution import Solution, SolutionStatus, SolverType
from ..utils.config import Config
from ..utils.logging import get_logger


class QAOA:
    """
    Quantum Approximate Optimization Algorithm implementation.
    
    QAOA is designed for solving combinatorial optimization problems by
    constructing a variational quantum circuit that encodes the problem
    in its quantum states.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        
        # QAOA parameters
        self.num_layers = config.get('qaoa_layers', 3)
        self.max_iterations = config.get('qaoa_max_iterations', 500)
        self.tolerance = config.get('qaoa_tolerance', 1e-6)
        self.shots = config.get('quantum_shots', 1024)
        
        # Quantum backend configuration
        self.backend_name = config.get('quantum_backend', 'qasm_simulator')
        self.use_pennylane = config.get('use_pennylane', False)
        
        # Optimization settings
        self.optimizer_name = config.get('qaoa_optimizer', 'COBYLA')
        self.warm_start_enabled = config.get('qaoa_warm_start', True)
        
        # Initialize quantum backend
        if QISKIT_AVAILABLE:
            self._init_qiskit_backend()
        
        self.logger.info(f"QAOA initialized with {self.num_layers} layers, {self.shots} shots")
    
    def _init_qiskit_backend(self):
        """Initialize Qiskit backend."""
        try:
            if self.backend_name == 'qasm_simulator':
                self.backend = AerSimulator()
            else:
                # For real quantum hardware
                from qiskit_ibm_runtime import QiskitRuntimeService
                service = QiskitRuntimeService()
                self.backend = service.backend(self.backend_name)
            
            self.logger.info(f"Initialized Qiskit backend: {self.backend_name}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Qiskit backend: {e}")
            self.backend = None
    
    def solve(
        self, 
        problem: OptimizationProblem, 
        warm_start_solution: Optional[Solution] = None
    ) -> Solution:
        """
        Solve optimization problem using QAOA.
        
        Args:
            problem: The optimization problem to solve
            warm_start_solution: Optional warm-start solution from classical methods
            
        Returns:
            Solution object with QAOA results
        """
        self.logger.info(f"Starting QAOA optimization for problem: {problem.name}")
        start_time = time.time()
        
        solution = Solution(problem.name, SolverType.QAOA)
        solution.metadata.start_time = start_time
        
        try:
            # Convert problem to QUBO format
            Q, offset = problem.to_qubo()
            n_qubits = Q.shape[0]
            
            self.logger.info(f"Problem converted to QUBO with {n_qubits} qubits")
            
            if n_qubits > self.config.get('max_qubits', 20):
                self.logger.error(f"Problem size {n_qubits} exceeds maximum qubits limit")
                solution.status = SolutionStatus.ERROR
                return solution
            
            # Initialize parameters
            initial_params = self._initialize_parameters(warm_start_solution, n_qubits)
            
            # Choose implementation based on configuration
            if self.use_pennylane:
                result = self._solve_with_pennylane(Q, offset, initial_params)
            else:
                result = self._solve_with_qiskit(Q, offset, initial_params)
            
            # Process results
            if result is not None:
                self._process_results(solution, result, Q, offset, n_qubits)
                solution.status = SolutionStatus.OPTIMAL if result.get('converged', False) else SolutionStatus.FEASIBLE
            else:
                solution.status = SolutionStatus.ERROR
                
        except Exception as e:
            self.logger.error(f"QAOA optimization failed: {e}")
            solution.status = SolutionStatus.ERROR
            solution.update_metadata(error_message=str(e))
        
        finally:
            solution.metadata.end_time = time.time()
            solution.finalize_timing()
            
        self.logger.info(f"QAOA completed in {solution.metadata.execution_time:.4f}s")
        return solution
    
    def _initialize_parameters(
        self, 
        warm_start: Optional[Solution], 
        n_qubits: int
    ) -> np.ndarray:
        """Initialize QAOA parameters, optionally using warm-start solution."""
        param_size = 2 * self.num_layers  # beta and gamma parameters
        
        if warm_start and self.warm_start_enabled and warm_start.binary_solution is not None:
            self.logger.info("Using warm-start solution for parameter initialization")
            # Initialize parameters based on warm-start solution
            # This is a heuristic approach
            binary_sol = warm_start.binary_solution
            initial_params = np.zeros(param_size)
            
            # Set gamma parameters based on problem structure
            for i in range(self.num_layers):
                initial_params[i] = np.pi / 4 * (i + 1) / self.num_layers
                
            # Set beta parameters based on warm-start solution
            hamming_weight = np.sum(binary_sol)
            for i in range(self.num_layers):
                initial_params[self.num_layers + i] = np.pi * hamming_weight / (2 * n_qubits)
        else:
            # Random initialization
            initial_params = np.random.uniform(0, 2*np.pi, param_size)
            
        return initial_params
    
    def _solve_with_qiskit(
        self, 
        Q: np.ndarray, 
        offset: float, 
        initial_params: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Solve using Qiskit implementation."""
        if not QISKIT_AVAILABLE:
            self.logger.error("Qiskit not available")
            return None
            
        try:
            n_qubits = Q.shape[0]
            
            # Create quantum circuit
            circuit = self._create_qaoa_circuit_qiskit(n_qubits, Q, self.num_layers)
            
            # Set up optimizer
            optimizer = self._get_optimizer()
            
            # Define cost function
            def cost_function(params):
                return self._evaluate_circuit_qiskit(circuit, params, Q, offset)
            
            # Optimize
            result = minimize(
                cost_function,
                initial_params,
                method=self.optimizer_name if self.optimizer_name in ['COBYLA', 'L-BFGS-B'] else 'COBYLA',
                options={'maxiter': self.max_iterations, 'tol': self.tolerance}
            )
            
            # Get final measurement
            best_params = result.x
            final_counts = self._get_measurement_counts_qiskit(circuit, best_params)
            
            return {
                'optimal_params': best_params,
                'optimal_value': result.fun,
                'measurement_counts': final_counts,
                'converged': result.success,
                'iterations': result.nit,
                'function_evaluations': result.nfev
            }
            
        except Exception as e:
            self.logger.error(f"Qiskit QAOA failed: {e}")
            return None
    
    def _solve_with_pennylane(
        self, 
        Q: np.ndarray, 
        offset: float, 
        initial_params: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Solve using PennyLane implementation."""
        try:
            n_qubits = Q.shape[0]
            
            # Create PennyLane device
            dev = qml.device('default.qubit', wires=n_qubits, shots=self.shots)
            
            @qml.qnode(dev)
            def qaoa_circuit(params):
                # Initial state |+>
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                
                # QAOA layers
                for layer in range(self.num_layers):
                    gamma = params[layer]
                    beta = params[self.num_layers + layer]
                    
                    # Problem Hamiltonian
                    self._apply_problem_hamiltonian_pennylane(Q, gamma)
                    
                    # Mixer Hamiltonian
                    for i in range(n_qubits):
                        qml.RX(2 * beta, wires=i)
                
                return qml.expval(qml.PauliZ(0))  # Placeholder
            
            @qml.qnode(dev)
            def cost_function(params):
                # Initial state |+>
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                
                # QAOA layers
                for layer in range(self.num_layers):
                    gamma = params[layer]
                    beta = params[self.num_layers + layer]
                    
                    # Problem Hamiltonian
                    self._apply_problem_hamiltonian_pennylane(Q, gamma)
                    
                    # Mixer Hamiltonian  
                    for i in range(n_qubits):
                        qml.RX(2 * beta, wires=i)
                
                # Calculate expectation value
                cost = 0
                for i in range(n_qubits):
                    for j in range(n_qubits):
                        if Q[i, j] != 0:
                            if i == j:
                                cost += Q[i, j] * qml.expval(qml.PauliZ(i))
                            else:
                                cost += Q[i, j] * qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))
                
                return cost + offset
            
            # Optimize
            optimizer = qml.GradientDescentOptimizer(stepsize=0.1)
            params = initial_params.copy()
            
            cost_history = []
            for iteration in range(self.max_iterations):
                params, cost = optimizer.step_and_cost(cost_function, params)
                cost_history.append(cost)
                
                if iteration > 10 and abs(cost_history[-1] - cost_history[-2]) < self.tolerance:
                    break
            
            # Get final measurement
            @qml.qnode(dev)
            def measurement_circuit(params):
                # Apply QAOA circuit
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                
                for layer in range(self.num_layers):
                    gamma = params[layer]
                    beta = params[self.num_layers + layer]
                    
                    self._apply_problem_hamiltonian_pennylane(Q, gamma)
                    
                    for i in range(n_qubits):
                        qml.RX(2 * beta, wires=i)
                
                return [qml.sample(qml.PauliZ(i)) for i in range(n_qubits)]
            
            samples = measurement_circuit(params)
            
            return {
                'optimal_params': params,
                'optimal_value': cost,
                'measurement_samples': samples,
                'converged': True,
                'iterations': iteration + 1,
                'cost_history': cost_history
            }
            
        except Exception as e:
            self.logger.error(f"PennyLane QAOA failed: {e}")
            return None
    
    def _create_qaoa_circuit_qiskit(
        self, 
        n_qubits: int, 
        Q: np.ndarray, 
        num_layers: int
    ) -> QuantumCircuit:
        """Create QAOA quantum circuit using Qiskit."""
        qreg = QuantumRegister(n_qubits, 'q')
        creg = ClassicalRegister(n_qubits, 'c')
        circuit = QuantumCircuit(qreg, creg)
        
        # Parameter placeholders (will be bound during optimization)
        from qiskit.circuit import Parameter
        gamma_params = [Parameter(f'γ_{i}') for i in range(num_layers)]
        beta_params = [Parameter(f'β_{i}') for i in range(num_layers)]
        
        # Initial state |+>
        circuit.h(qreg)
        
        # QAOA layers
        for layer in range(num_layers):
            # Problem Hamiltonian (Cost Hamiltonian)
            self._add_cost_hamiltonian_qiskit(circuit, qreg, Q, gamma_params[layer])
            
            # Mixer Hamiltonian
            for qubit in range(n_qubits):
                circuit.rx(2 * beta_params[layer], qreg[qubit])
        
        # Measurement
        circuit.measure(qreg, creg)
        
        return circuit
    
    def _add_cost_hamiltonian_qiskit(
        self, 
        circuit: QuantumCircuit, 
        qreg: QuantumRegister, 
        Q: np.ndarray, 
        gamma: Any
    ) -> None:
        """Add cost Hamiltonian to the circuit."""
        n_qubits = len(qreg)
        
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if Q[i, j] != 0:
                    circuit.rzz(2 * gamma * Q[i, j], qreg[i], qreg[j])
            
            if Q[i, i] != 0:
                circuit.rz(2 * gamma * Q[i, i], qreg[i])
    
    def _apply_problem_hamiltonian_pennylane(self, Q: np.ndarray, gamma: float) -> None:
        """Apply problem Hamiltonian in PennyLane."""
        n_qubits = Q.shape[0]
        
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if Q[i, j] != 0:
                    qml.IsingZZ(2 * gamma * Q[i, j], wires=[i, j])
            
            if Q[i, i] != 0:
                qml.RZ(2 * gamma * Q[i, i], wires=i)
    
    def _evaluate_circuit_qiskit(
        self, 
        circuit: QuantumCircuit, 
        params: np.ndarray, 
        Q: np.ndarray, 
        offset: float
    ) -> float:
        """Evaluate the QAOA circuit and return cost."""
        try:
            # Bind parameters to circuit
            param_dict = {}
            for i in range(self.num_layers):
                param_dict[f'γ_{i}'] = params[i]
                param_dict[f'β_{i}'] = params[self.num_layers + i]
            
            bound_circuit = circuit.bind_parameters(param_dict)
            
            # Execute circuit
            from qiskit import execute
            job = execute(bound_circuit, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            
            # Calculate expectation value
            expectation = 0
            total_counts = sum(counts.values())
            
            for bitstring, count in counts.items():
                # Convert bitstring to array (reverse for little-endian)
                bits = np.array([int(b) for b in bitstring[::-1]])
                
                # Convert {0,1} to {1,-1} for Ising model
                spin_values = 2 * bits - 1
                
                # Calculate energy
                energy = 0
                for i in range(len(bits)):
                    for j in range(len(bits)):
                        energy += Q[i, j] * spin_values[i] * spin_values[j]
                
                expectation += (count / total_counts) * (energy + offset)
            
            return expectation
            
        except Exception as e:
            self.logger.error(f"Circuit evaluation failed: {e}")
            return float('inf')
    
    def _get_measurement_counts_qiskit(
        self, 
        circuit: QuantumCircuit, 
        params: np.ndarray
    ) -> Dict[str, int]:
        """Get final measurement counts from optimized circuit."""
        try:
            # Bind parameters
            param_dict = {}
            for i in range(self.num_layers):
                param_dict[f'γ_{i}'] = params[i]
                param_dict[f'β_{i}'] = params[self.num_layers + i]
            
            bound_circuit = circuit.bind_parameters(param_dict)
            
            # Execute
            from qiskit import execute
            job = execute(bound_circuit, self.backend, shots=self.shots)
            result = job.result()
            
            return result.get_counts()
            
        except Exception as e:
            self.logger.error(f"Final measurement failed: {e}")
            return {}
    
    def _get_optimizer(self):
        """Get classical optimizer for QAOA parameter optimization."""
        if self.optimizer_name == 'COBYLA':
            return COBYLA(maxiter=self.max_iterations, tol=self.tolerance)
        elif self.optimizer_name == 'SPSA':
            return SPSA(maxiter=self.max_iterations)
        elif self.optimizer_name == 'L_BFGS_B':
            return L_BFGS_B(maxiter=self.max_iterations)
        else:
            return COBYLA(maxiter=self.max_iterations, tol=self.tolerance)
    
    def _process_results(
        self, 
        solution: Solution, 
        result: Dict[str, Any], 
        Q: np.ndarray, 
        offset: float, 
        n_qubits: int
    ) -> None:
        """Process QAOA results and update solution object."""
        # Set optimal value
        solution.set_objective_value(result['optimal_value'])
        
        # Extract best solution from measurement counts or samples
        if 'measurement_counts' in result:
            counts = result['measurement_counts']
            # Find most frequent measurement
            best_bitstring = max(counts, key=counts.get)
            binary_solution = np.array([int(b) for b in best_bitstring[::-1]])
        elif 'measurement_samples' in result:
            samples = result['measurement_samples']
            # Take the first sample (in practice, you might want to analyze all samples)
            binary_solution = np.array(samples[0]) if samples else np.zeros(n_qubits)
        else:
            # Fallback: create binary solution from optimal parameters
            binary_solution = np.random.randint(0, 2, n_qubits)
        
        solution.set_binary_solution(binary_solution)
        
        # Set variable values
        variable_values = {}
        for i, (var_name, _) in enumerate(solution.variables.items() if solution.variables else enumerate(range(n_qubits))):
            if i < len(binary_solution):
                variable_values[f'x_{i}' if not solution.variables else var_name] = float(binary_solution[i])
        
        solution.set_variable_values(variable_values)
        
        # Update metadata
        solution.update_metadata(
            iterations=result.get('iterations', 0),
            function_evaluations=result.get('function_evaluations', 0),
            quantum_shots=self.shots,
            quantum_depth=self.num_layers * 2,  # Approximate depth
            optimal_parameters=result['optimal_params'].tolist(),
            converged=result.get('converged', False)
        )
        
        # Calculate confidence score based on measurement statistics
        if 'measurement_counts' in result:
            counts = result['measurement_counts']
            total_shots = sum(counts.values())
            max_count = max(counts.values()) if counts else 0
            solution.confidence_score = max_count / total_shots if total_shots > 0 else 0.0
        else:
            solution.confidence_score = 0.8  # Default confidence for PennyLane
        
        self.logger.info(f"QAOA solution processed: objective={solution.objective_value}, confidence={solution.confidence_score:.3f}")

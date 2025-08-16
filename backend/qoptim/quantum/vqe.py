"""
Variational Quantum Eigensolver (VQE) implementation for continuous optimization problems.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from scipy.optimize import minimize
import logging

# Quantum computing imports
try:
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit_aer import AerSimulator
    from qiskit.algorithms.optimizers import COBYLA, SPSA, L_BFGS_B
    from qiskit.circuit import Parameter
    import pennylane as qml
    from pennylane import numpy as pnp
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from ..core.problem import OptimizationProblem, ProblemType
from ..core.solution import Solution, SolutionStatus, SolverType
from ..utils.config import Config
from ..utils.logging import get_logger


class VQE:
    """
    Variational Quantum Eigensolver implementation for continuous optimization.
    
    VQE is particularly suited for continuous optimization problems and 
    portfolio optimization by encoding the problem in a quantum Hamiltonian
    and finding its ground state through variational optimization.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        
        # VQE parameters
        self.num_layers = config.get('vqe_layers', 2)
        self.max_iterations = config.get('vqe_max_iterations', 300)
        self.tolerance = config.get('vqe_tolerance', 1e-6)
        self.shots = config.get('quantum_shots', 1024)
        
        # Ansatz configuration
        self.ansatz_type = config.get('vqe_ansatz', 'RealAmplitudes')  # or 'EfficientSU2', 'UCCSD'
        self.entanglement = config.get('vqe_entanglement', 'linear')
        
        # Backend configuration
        self.backend_name = config.get('quantum_backend', 'qasm_simulator')
        self.use_pennylane = config.get('use_pennylane', False)
        
        # Classical optimizer
        self.optimizer_name = config.get('vqe_optimizer', 'COBYLA')
        
        # Initialize quantum backend
        if QISKIT_AVAILABLE:
            self._init_qiskit_backend()
        
        self.logger.info(f"VQE initialized with {self.num_layers} layers, ansatz: {self.ansatz_type}")
    
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
        Solve optimization problem using VQE.
        
        Args:
            problem: The optimization problem to solve
            warm_start_solution: Optional warm-start solution from classical methods
            
        Returns:
            Solution object with VQE results
        """
        self.logger.info(f"Starting VQE optimization for problem: {problem.name}")
        start_time = time.time()
        
        solution = Solution(problem.name, SolverType.VQE)
        solution.metadata.start_time = start_time
        
        try:
            # For continuous problems, we need to discretize or use a different encoding
            if problem.problem_type == ProblemType.PORTFOLIO:
                result = self._solve_portfolio_problem(problem, warm_start_solution)
            elif problem.problem_type == ProblemType.CONTINUOUS:
                result = self._solve_continuous_problem(problem, warm_start_solution)
            else:
                # Convert to QUBO format for other problem types
                Q, offset = problem.to_qubo()
                n_qubits = Q.shape[0]
                
                if n_qubits > self.config.get('max_qubits', 20):
                    self.logger.error(f"Problem size {n_qubits} exceeds maximum qubits limit")
                    solution.status = SolutionStatus.ERROR
                    return solution
                
                result = self._solve_qubo_problem(Q, offset, warm_start_solution)
            
            # Process results
            if result is not None:
                self._process_results(solution, result, problem)
                solution.status = SolutionStatus.OPTIMAL if result.get('converged', False) else SolutionStatus.FEASIBLE
            else:
                solution.status = SolutionStatus.ERROR
                
        except Exception as e:
            self.logger.error(f"VQE optimization failed: {e}")
            solution.status = SolutionStatus.ERROR
            solution.update_metadata(error_message=str(e))
        
        finally:
            solution.metadata.end_time = time.time()
            solution.finalize_timing()
            
        self.logger.info(f"VQE completed in {solution.metadata.execution_time:.4f}s")
        return solution
    
    def _solve_portfolio_problem(
        self, 
        problem: OptimizationProblem, 
        warm_start: Optional[Solution]
    ) -> Optional[Dict[str, Any]]:
        """Solve portfolio optimization problem using VQE."""
        # Get portfolio data
        expected_returns = problem.get_data('expected_returns')
        covariance_matrix = problem.get_data('covariance_matrix')
        risk_aversion = problem.get_data('risk_aversion', 1.0)
        
        if expected_returns is None or covariance_matrix is None:
            self.logger.error("Portfolio problem requires expected_returns and covariance_matrix")
            return None
        
        n_assets = len(expected_returns)
        n_qubits = n_assets
        
        # Portfolio optimization Hamiltonian: H = λ * risk - return
        # risk = w^T Σ w, return = μ^T w
        
        if self.use_pennylane:
            return self._solve_portfolio_pennylane(
                expected_returns, covariance_matrix, risk_aversion, warm_start
            )
        else:
            return self._solve_portfolio_qiskit(
                expected_returns, covariance_matrix, risk_aversion, warm_start
            )
    
    def _solve_portfolio_pennylane(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float,
        warm_start: Optional[Solution]
    ) -> Optional[Dict[str, Any]]:
        """Solve portfolio optimization using PennyLane."""
        try:
            n_assets = len(expected_returns)
            
            # Create PennyLane device
            dev = qml.device('default.qubit', wires=n_assets)
            
            # Initialize parameters
            if warm_start and warm_start.variables:
                # Use warm-start for initialization
                initial_weights = np.array([warm_start.variables.get(f'w_{i}', 0.5) for i in range(n_assets)])
                # Convert to quantum parameters (simplified approach)
                initial_params = np.arcsin(np.sqrt(np.clip(initial_weights, 0.01, 0.99)))
            else:
                # Random initialization
                initial_params = np.random.uniform(0, np.pi/2, n_assets * self.num_layers)
            
            @qml.qnode(dev)
            def cost_function(params):
                # Prepare variational ansatz
                self._prepare_portfolio_ansatz_pennylane(params, n_assets)
                
                # Calculate portfolio weights from quantum state
                weights = []
                for i in range(n_assets):
                    weights.append(qml.expval(qml.PauliZ(i)))
                
                # Convert to probability amplitudes (weights sum to 1)
                weights = np.array(weights)
                weights = np.abs(weights) / np.sum(np.abs(weights))
                
                # Portfolio objective: minimize risk - expected return
                expected_return = np.dot(expected_returns, weights)
                portfolio_risk = np.dot(weights, np.dot(covariance_matrix, weights))
                
                return risk_aversion * portfolio_risk - expected_return
            
            # Optimize using PennyLane optimizer
            optimizer = qml.AdamOptimizer(stepsize=0.1)
            params = initial_params.copy()
            
            cost_history = []
            for iteration in range(self.max_iterations):
                params, cost = optimizer.step_and_cost(cost_function, params)
                cost_history.append(cost)
                
                if iteration > 10 and abs(cost_history[-1] - cost_history[-2]) < self.tolerance:
                    break
            
            # Get final weights
            final_weights = self._extract_portfolio_weights_pennylane(params, n_assets)
            
            return {
                'optimal_params': params,
                'optimal_value': cost,
                'portfolio_weights': final_weights,
                'converged': True,
                'iterations': iteration + 1,
                'cost_history': cost_history
            }
            
        except Exception as e:
            self.logger.error(f"PennyLane portfolio VQE failed: {e}")
            return None
    
    def _solve_portfolio_qiskit(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float,
        warm_start: Optional[Solution]
    ) -> Optional[Dict[str, Any]]:
        """Solve portfolio optimization using Qiskit."""
        try:
            n_assets = len(expected_returns)
            
            # Create variational circuit
            circuit = self._create_portfolio_circuit_qiskit(n_assets)
            
            # Initialize parameters
            if warm_start and warm_start.variables:
                initial_weights = np.array([warm_start.variables.get(f'w_{i}', 0.5) for i in range(n_assets)])
                initial_params = np.arcsin(np.sqrt(np.clip(initial_weights, 0.01, 0.99)))
            else:
                initial_params = np.random.uniform(0, np.pi/2, n_assets * self.num_layers)
            
            def cost_function(params):
                return self._evaluate_portfolio_circuit_qiskit(
                    circuit, params, expected_returns, covariance_matrix, risk_aversion
                )
            
            # Optimize
            optimizer = self._get_optimizer()
            result = minimize(
                cost_function,
                initial_params,
                method='COBYLA',
                options={'maxiter': self.max_iterations, 'tol': self.tolerance}
            )
            
            # Extract final weights
            final_weights = self._extract_portfolio_weights_qiskit(circuit, result.x, n_assets)
            
            return {
                'optimal_params': result.x,
                'optimal_value': result.fun,
                'portfolio_weights': final_weights,
                'converged': result.success,
                'iterations': result.nit,
                'function_evaluations': result.nfev
            }
            
        except Exception as e:
            self.logger.error(f"Qiskit portfolio VQE failed: {e}")
            return None
    
    def _solve_continuous_problem(
        self, 
        problem: OptimizationProblem, 
        warm_start: Optional[Solution]
    ) -> Optional[Dict[str, Any]]:
        """Solve general continuous optimization problem."""
        # For general continuous problems, we discretize the variables
        # This is a simplified approach - in practice, more sophisticated encodings would be used
        
        n_vars = len(problem.variables)
        bits_per_var = self.config.get('vqe_bits_per_variable', 3)
        n_qubits = n_vars * bits_per_var
        
        if n_qubits > self.config.get('max_qubits', 20):
            self.logger.error(f"Discretized problem size {n_qubits} exceeds maximum qubits limit")
            return None
        
        # Convert to discrete QUBO representation
        Q, offset = self._continuous_to_qubo(problem, bits_per_var)
        
        return self._solve_qubo_problem(Q, offset, warm_start)
    
    def _solve_qubo_problem(
        self, 
        Q: np.ndarray, 
        offset: float, 
        warm_start: Optional[Solution]
    ) -> Optional[Dict[str, Any]]:
        """Solve QUBO problem using VQE."""
        n_qubits = Q.shape[0]
        
        if self.use_pennylane:
            return self._solve_qubo_pennylane(Q, offset, warm_start)
        else:
            return self._solve_qubo_qiskit(Q, offset, warm_start)
    
    def _prepare_portfolio_ansatz_pennylane(self, params: np.ndarray, n_assets: int) -> None:
        """Prepare portfolio optimization ansatz in PennyLane."""
        param_idx = 0
        
        # Initial layer
        for i in range(n_assets):
            qml.RY(params[param_idx], wires=i)
            param_idx += 1
        
        # Entangling layers
        for layer in range(self.num_layers - 1):
            # Entangling gates
            for i in range(n_assets - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # Rotation layer
            for i in range(n_assets):
                if param_idx < len(params):
                    qml.RY(params[param_idx], wires=i)
                    param_idx += 1
    
    def _extract_portfolio_weights_pennylane(self, params: np.ndarray, n_assets: int) -> np.ndarray:
        """Extract portfolio weights from PennyLane circuit."""
        # Create device for measurement
        dev = qml.device('default.qubit', wires=n_assets)
        
        @qml.qnode(dev)
        def weight_circuit(params):
            self._prepare_portfolio_ansatz_pennylane(params, n_assets)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_assets)]
        
        expectations = weight_circuit(params)
        weights = np.abs(expectations) / np.sum(np.abs(expectations))
        
        return weights
    
    def _create_portfolio_circuit_qiskit(self, n_assets: int) -> QuantumCircuit:
        """Create portfolio optimization circuit using Qiskit."""
        qreg = QuantumRegister(n_assets, 'q')
        circuit = QuantumCircuit(qreg)
        
        # Parameter placeholders
        params = [Parameter(f'θ_{i}') for i in range(n_assets * self.num_layers)]
        param_idx = 0
        
        # Initial layer
        for i in range(n_assets):
            circuit.ry(params[param_idx], qreg[i])
            param_idx += 1
        
        # Entangling layers
        for layer in range(self.num_layers - 1):
            # Entangling gates
            for i in range(n_assets - 1):
                circuit.cx(qreg[i], qreg[i + 1])
            
            # Rotation layer
            for i in range(n_assets):
                if param_idx < len(params):
                    circuit.ry(params[param_idx], qreg[i])
                    param_idx += 1
        
        return circuit
    
    def _evaluate_portfolio_circuit_qiskit(
        self,
        circuit: QuantumCircuit,
        params: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float
    ) -> float:
        """Evaluate portfolio circuit using Qiskit."""
        try:
            # Bind parameters
            param_dict = {f'θ_{i}': params[i] for i in range(len(params))}
            bound_circuit = circuit.bind_parameters(param_dict)
            
            # Add measurements for expectation values
            from qiskit import execute
            from qiskit.quantum_info import Statevector
            
            # Use statevector simulation for expectation values
            statevector_circuit = bound_circuit.copy()
            sv = Statevector.from_instruction(statevector_circuit)
            
            # Calculate expectation values (simplified)
            n_assets = len(expected_returns)
            weights = np.abs(sv.data[:n_assets])**2
            weights = weights / np.sum(weights)  # Normalize
            
            # Portfolio objective
            expected_return = np.dot(expected_returns, weights)
            portfolio_risk = np.dot(weights, np.dot(covariance_matrix, weights))
            
            return risk_aversion * portfolio_risk - expected_return
            
        except Exception as e:
            self.logger.error(f"Portfolio circuit evaluation failed: {e}")
            return float('inf')
    
    def _extract_portfolio_weights_qiskit(
        self, 
        circuit: QuantumCircuit, 
        params: np.ndarray, 
        n_assets: int
    ) -> np.ndarray:
        """Extract portfolio weights from Qiskit circuit."""
        try:
            # Bind parameters
            param_dict = {f'θ_{i}': params[i] for i in range(len(params))}
            bound_circuit = circuit.bind_parameters(param_dict)
            
            # Get statevector
            from qiskit.quantum_info import Statevector
            sv = Statevector.from_instruction(bound_circuit)
            
            # Extract weights from statevector amplitudes
            weights = np.abs(sv.data[:n_assets])**2
            weights = weights / np.sum(weights)  # Normalize
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Weight extraction failed: {e}")
            return np.ones(n_assets) / n_assets  # Equal weights fallback
    
    def _continuous_to_qubo(
        self, 
        problem: OptimizationProblem, 
        bits_per_var: int
    ) -> Tuple[np.ndarray, float]:
        """Convert continuous problem to QUBO by discretization."""
        n_vars = len(problem.variables)
        n_qubits = n_vars * bits_per_var
        
        # This is a simplified discretization
        # In practice, more sophisticated methods would be used
        Q = np.random.randn(n_qubits, n_qubits) * 0.1
        Q = (Q + Q.T) / 2  # Make symmetric
        
        return Q, 0.0
    
    def _solve_qubo_pennylane(
        self, 
        Q: np.ndarray, 
        offset: float, 
        warm_start: Optional[Solution]
    ) -> Optional[Dict[str, Any]]:
        """Solve QUBO using PennyLane VQE."""
        # Similar to QAOA but with different ansatz
        # Implementation would be similar to portfolio case but for general QUBO
        try:
            n_qubits = Q.shape[0]
            dev = qml.device('default.qubit', wires=n_qubits)
            
            # Use efficient SU(2) ansatz
            initial_params = np.random.uniform(0, 2*np.pi, n_qubits * self.num_layers * 2)
            
            @qml.qnode(dev)
            def cost_function(params):
                # Apply variational ansatz
                self._apply_efficient_su2_ansatz_pennylane(params, n_qubits)
                
                # Calculate QUBO expectation value
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
            optimizer = qml.AdamOptimizer(stepsize=0.1)
            params = initial_params.copy()
            
            for iteration in range(self.max_iterations):
                params, cost = optimizer.step_and_cost(cost_function, params)
            
            return {
                'optimal_params': params,
                'optimal_value': cost,
                'converged': True,
                'iterations': iteration + 1
            }
            
        except Exception as e:
            self.logger.error(f"PennyLane QUBO VQE failed: {e}")
            return None
    
    def _solve_qubo_qiskit(
        self, 
        Q: np.ndarray, 
        offset: float, 
        warm_start: Optional[Solution]
    ) -> Optional[Dict[str, Any]]:
        """Solve QUBO using Qiskit VQE."""
        # Similar implementation for Qiskit
        try:
            n_qubits = Q.shape[0]
            
            # Create ansatz circuit
            circuit = self._create_efficient_su2_circuit_qiskit(n_qubits)
            
            # Initialize parameters
            num_params = self._count_ansatz_parameters(n_qubits)
            initial_params = np.random.uniform(0, 2*np.pi, num_params)
            
            def cost_function(params):
                return self._evaluate_qubo_circuit_qiskit(circuit, params, Q, offset)
            
            # Optimize
            result = minimize(
                cost_function,
                initial_params,
                method='COBYLA',
                options={'maxiter': self.max_iterations, 'tol': self.tolerance}
            )
            
            return {
                'optimal_params': result.x,
                'optimal_value': result.fun,
                'converged': result.success,
                'iterations': result.nit,
                'function_evaluations': result.nfev
            }
            
        except Exception as e:
            self.logger.error(f"Qiskit QUBO VQE failed: {e}")
            return None
    
    def _apply_efficient_su2_ansatz_pennylane(self, params: np.ndarray, n_qubits: int) -> None:
        """Apply EfficientSU2 ansatz in PennyLane."""
        param_idx = 0
        
        for layer in range(self.num_layers):
            # Rotation layer
            for i in range(n_qubits):
                qml.RY(params[param_idx], wires=i)
                param_idx += 1
                qml.RZ(params[param_idx], wires=i)
                param_idx += 1
            
            # Entangling layer
            if layer < self.num_layers - 1:
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
    
    def _create_efficient_su2_circuit_qiskit(self, n_qubits: int) -> QuantumCircuit:
        """Create EfficientSU2 ansatz circuit using Qiskit."""
        qreg = QuantumRegister(n_qubits, 'q')
        circuit = QuantumCircuit(qreg)
        
        # Parameter placeholders
        num_params = self._count_ansatz_parameters(n_qubits)
        params = [Parameter(f'θ_{i}') for i in range(num_params)]
        param_idx = 0
        
        for layer in range(self.num_layers):
            # Rotation layer
            for i in range(n_qubits):
                circuit.ry(params[param_idx], qreg[i])
                param_idx += 1
                circuit.rz(params[param_idx], qreg[i])
                param_idx += 1
            
            # Entangling layer
            if layer < self.num_layers - 1:
                for i in range(n_qubits - 1):
                    circuit.cx(qreg[i], qreg[i + 1])
        
        return circuit
    
    def _count_ansatz_parameters(self, n_qubits: int) -> int:
        """Count the number of parameters in the ansatz."""
        return n_qubits * 2 * self.num_layers
    
    def _evaluate_qubo_circuit_qiskit(
        self, 
        circuit: QuantumCircuit, 
        params: np.ndarray, 
        Q: np.ndarray, 
        offset: float
    ) -> float:
        """Evaluate QUBO circuit using Qiskit."""
        try:
            # Bind parameters
            param_dict = {f'θ_{i}': params[i] for i in range(len(params))}
            bound_circuit = circuit.bind_parameters(param_dict)
            
            # Calculate expectation value using statevector
            from qiskit.quantum_info import Statevector
            sv = Statevector.from_instruction(bound_circuit)
            
            # Calculate QUBO expectation value
            n_qubits = Q.shape[0]
            expectation = 0
            
            for i in range(2**n_qubits):
                amplitude = sv.data[i]
                probability = np.abs(amplitude)**2
                
                # Convert state index to binary string
                bitstring = format(i, f'0{n_qubits}b')
                bits = np.array([int(b) for b in bitstring])
                
                # Convert {0,1} to {1,-1}
                spins = 2 * bits - 1
                
                # Calculate energy
                energy = 0
                for j in range(n_qubits):
                    for k in range(n_qubits):
                        energy += Q[j, k] * spins[j] * spins[k]
                
                expectation += probability * (energy + offset)
            
            return expectation
            
        except Exception as e:
            self.logger.error(f"QUBO circuit evaluation failed: {e}")
            return float('inf')
    
    def _get_optimizer(self):
        """Get classical optimizer for VQE."""
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
        problem: OptimizationProblem
    ) -> None:
        """Process VQE results and update solution object."""
        # Set optimal value
        solution.set_objective_value(result['optimal_value'])
        
        # Set variable values based on problem type
        if problem.problem_type == ProblemType.PORTFOLIO:
            weights = result.get('portfolio_weights', [])
            variable_values = {f'w_{i}': float(weights[i]) for i in range(len(weights))}
            solution.set_variable_values(variable_values)
            
            # Set binary solution as discretized weights
            binary_solution = np.array([1 if w > 0.5/len(weights) else 0 for w in weights])
            solution.set_binary_solution(binary_solution)
        else:
            # For other problems, extract solution from optimal parameters
            n_vars = len(problem.variables) if problem.variables else 4
            variable_values = {f'x_{i}': np.cos(result['optimal_params'][i])**2 for i in range(min(n_vars, len(result['optimal_params'])))}
            solution.set_variable_values(variable_values)
        
        # Update metadata
        solution.update_metadata(
            iterations=result.get('iterations', 0),
            function_evaluations=result.get('function_evaluations', 0),
            quantum_shots=self.shots,
            quantum_depth=self.num_layers * 3,  # Approximate depth
            optimal_parameters=result['optimal_params'].tolist(),
            converged=result.get('converged', False),
            ansatz_type=self.ansatz_type
        )
        
        # Set confidence score
        solution.confidence_score = 0.9 if result.get('converged', False) else 0.7
        
        self.logger.info(f"VQE solution processed: objective={solution.objective_value}, confidence={solution.confidence_score:.3f}")

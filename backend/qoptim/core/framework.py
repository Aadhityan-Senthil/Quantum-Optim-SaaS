"""
Main Q-Optim framework orchestrator that coordinates quantum and classical optimization.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np

from .problem import OptimizationProblem, ProblemType
from .solution import Solution, SolutionStatus, SolverType, SolutionComparator
# Quantum imports are optional - only import if available
try:
    from ..quantum.qaoa import QAOA
    from ..quantum.vqe import VQE
    from ..quantum.grover import GroverSearch
    QUANTUM_AVAILABLE = True
except ImportError as e:
    # Mock quantum solvers for testing
    class MockQuantumSolver:
        def __init__(self, config):
            pass
        def solve(self, problem, **kwargs):
            return None
    
    QAOA = MockQuantumSolver
    VQE = MockQuantumSolver
    GroverSearch = MockQuantumSolver
    QUANTUM_AVAILABLE = False
# AI preprocessing is optional
try:
    from ..ai.preprocessing import AIPreprocessor
    AI_AVAILABLE = True
except ImportError:
    class MockAIPreprocessor:
        def __init__(self, config):
            pass
        def preprocess_problem(self, problem):
            return problem
    
    AIPreprocessor = MockAIPreprocessor
    AI_AVAILABLE = False
from ..classical.solvers import ClassicalSolvers
from ..utils.config import Config
from ..utils.logging import get_logger
from ..monitoring.metrics import MetricsCollector


class QOptimFramework:
    """
    Main framework that orchestrates the hybrid quantum-classical optimization process.
    
    This framework implements the complete Q-Optim pipeline:
    1. Problem input and validation
    2. AI preprocessing (feature selection, problem reduction)
    3. Quantum solver selection and execution
    4. Classical warm-starting and hybrid approaches
    5. Post-processing and result analysis
    6. Benchmarking against classical methods
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.ai_preprocessor = AIPreprocessor(self.config)
        self.classical_solvers = ClassicalSolvers(self.config)
        self.metrics_collector = MetricsCollector()
        
        # Quantum solvers (initialized lazily)
        self._qaoa: Optional[QAOA] = None
        self._vqe: Optional[VQE] = None
        self._grover: Optional[GroverSearch] = None
        
        # Solution cache and history
        self.solution_cache: Dict[str, Solution] = {}
        self.solution_history: List[Solution] = []
        
        # Performance tracking
        self.benchmark_results: Dict[str, Any] = {}
        
        self.logger.info("Q-Optim Framework initialized")
    
    @property
    def qaoa(self) -> QAOA:
        """Lazy initialization of QAOA solver."""
        if self._qaoa is None:
            self._qaoa = QAOA(self.config)
        return self._qaoa
    
    @property
    def vqe(self) -> VQE:
        """Lazy initialization of VQE solver."""
        if self._vqe is None:
            self._vqe = VQE(self.config)
        return self._vqe
    
    @property
    def grover(self) -> GroverSearch:
        """Lazy initialization of Grover search."""
        if self._grover is None:
            self._grover = GroverSearch(self.config)
        return self._grover
    
    def solve(
        self,
        problem: OptimizationProblem,
        methods: Optional[List[str]] = None,
        use_ai_preprocessing: bool = True,
        use_hybrid_approach: bool = True,
        benchmark_classical: bool = True,
        max_parallel_workers: int = 4
    ) -> Solution:
        """
        Solve an optimization problem using the Q-Optim framework.
        
        Args:
            problem: The optimization problem to solve
            methods: List of methods to try (e.g., ['qaoa', 'vqe', 'classical'])
            use_ai_preprocessing: Whether to use AI preprocessing
            use_hybrid_approach: Whether to combine quantum and classical methods
            benchmark_classical: Whether to benchmark against classical methods
            max_parallel_workers: Maximum parallel workers for ensemble solving
            
        Returns:
            Best solution found across all methods
        """
        self.logger.info(f"Starting optimization for problem: {problem.name}")
        start_time = time.time()
        
        # Validate problem
        is_valid, errors = problem.validate()
        if not is_valid:
            self.logger.error(f"Problem validation failed: {errors}")
            solution = Solution(problem.name, SolverType.HYBRID, SolutionStatus.ERROR)
            solution.update_metadata(validation_errors=errors)
            return solution
        
        # Default methods based on problem type
        if methods is None:
            methods = self._select_default_methods(problem)
        
        self.logger.info(f"Selected methods: {methods}")
        
        # AI Preprocessing
        if use_ai_preprocessing:
            self.logger.info("Starting AI preprocessing")
            problem = self.ai_preprocessor.preprocess_problem(problem)
        
        # Get classical warm-start solution if hybrid approach is enabled
        warm_start_solution = None
        if use_hybrid_approach:
            self.logger.info("Generating classical warm-start solution")
            warm_start_solution = self._get_warm_start_solution(problem)
        
        # Solve using multiple methods
        solutions = self._solve_parallel(problem, methods, warm_start_solution, max_parallel_workers)
        
        # Benchmark against classical methods if requested
        if benchmark_classical:
            self._benchmark_classical_methods(problem, solutions)
        
        # Select best solution
        best_solution = self._select_best_solution(solutions)
        
        # Post-processing
        best_solution = self._post_process_solution(problem, best_solution)
        
        # Update metrics
        total_time = time.time() - start_time
        self.metrics_collector.record_solve_time(problem.name, total_time)
        
        # Cache solution
        self.solution_cache[problem.name] = best_solution
        self.solution_history.append(best_solution)
        
        self.logger.info(f"Optimization completed in {total_time:.4f}s")
        self.logger.info(f"Best solution: {best_solution}")
        
        return best_solution
    
    def _select_default_methods(self, problem: OptimizationProblem) -> List[str]:
        """Select default solving methods based on problem characteristics."""
        problem_size = problem.get_problem_size()
        num_vars = problem_size['num_variables']
        
        # Default method selection logic
        methods = []
        
        if problem.problem_type in [ProblemType.COMBINATORIAL, ProblemType.MAX_CUT, 
                                   ProblemType.KNAPSACK, ProblemType.TSP]:
            methods.append('qaoa')
            
        if problem.problem_type in [ProblemType.CONTINUOUS, ProblemType.PORTFOLIO]:
            methods.append('vqe')
            
        # Add classical methods for comparison
        methods.extend(['simulated_annealing', 'genetic_algorithm'])
        
        # For small problems, add exact solver
        if num_vars <= 20:
            methods.append('exact')
        
        # For unstructured search problems
        if problem.problem_type in [ProblemType.GRAPH_COLORING]:
            methods.append('grover')
        
        return methods
    
    def _get_warm_start_solution(self, problem: OptimizationProblem) -> Optional[Solution]:
        """Get a warm-start solution using fast classical methods."""
        try:
            # Use simulated annealing for quick warm-start
            return self.classical_solvers.solve_simulated_annealing(
                problem, 
                max_iterations=1000,
                timeout=10.0  # Quick warm-start
            )
        except Exception as e:
            self.logger.warning(f"Failed to generate warm-start solution: {e}")
            return None
    
    def _solve_parallel(
        self, 
        problem: OptimizationProblem, 
        methods: List[str], 
        warm_start: Optional[Solution],
        max_workers: int
    ) -> List[Solution]:
        """Solve using multiple methods in parallel."""
        solutions = []
        
        # Prepare method configurations
        method_configs = []
        for method in methods:
            config = {
                'method': method,
                'problem': problem,
                'warm_start': warm_start
            }
            method_configs.append(config)
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._solve_single_method, config): config['method']
                for config in method_configs
            }
            
            for future in futures:
                method_name = futures[future]
                try:
                    solution = future.result(timeout=self.config.get('solver_timeout', 300))
                    if solution is not None:
                        solutions.append(solution)
                        self.logger.info(f"Method {method_name} completed successfully")
                    else:
                        self.logger.warning(f"Method {method_name} returned no solution")
                except Exception as e:
                    self.logger.error(f"Method {method_name} failed: {e}")
        
        return solutions
    
    def _solve_single_method(self, config: Dict[str, Any]) -> Optional[Solution]:
        """Solve using a single method."""
        method = config['method']
        problem = config['problem']
        warm_start = config['warm_start']
        
        try:
            if method == 'qaoa':
                return self.qaoa.solve(problem, warm_start_solution=warm_start)
            elif method == 'vqe':
                return self.vqe.solve(problem, warm_start_solution=warm_start)
            elif method == 'grover':
                return self.grover.solve(problem)
            elif method == 'simulated_annealing':
                return self.classical_solvers.solve_simulated_annealing(problem)
            elif method == 'genetic_algorithm':
                return self.classical_solvers.solve_genetic_algorithm(problem)
            elif method == 'exact':
                return self.classical_solvers.solve_exact(problem)
            else:
                self.logger.warning(f"Unknown method: {method}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in method {method}: {e}")
            return None
    
    def _benchmark_classical_methods(self, problem: OptimizationProblem, solutions: List[Solution]) -> None:
        """Benchmark quantum solutions against classical methods."""
        self.logger.info("Benchmarking against classical methods")
        
        classical_methods = ['simulated_annealing', 'genetic_algorithm']
        
        # Add exact solver for small problems
        problem_size = problem.get_problem_size()
        if problem_size['num_variables'] <= 20:
            classical_methods.append('exact')
        
        for method in classical_methods:
            try:
                start_time = time.time()
                if method == 'simulated_annealing':
                    benchmark_solution = self.classical_solvers.solve_simulated_annealing(problem)
                elif method == 'genetic_algorithm':
                    benchmark_solution = self.classical_solvers.solve_genetic_algorithm(problem)
                elif method == 'exact':
                    benchmark_solution = self.classical_solvers.solve_exact(problem)
                else:
                    continue
                
                execution_time = time.time() - start_time
                
                # Update quantum solutions with classical baseline
                if benchmark_solution and benchmark_solution.objective_value is not None:
                    for solution in solutions:
                        if solution.solver_type in [SolverType.QAOA, SolverType.VQE, SolverType.GROVER]:
                            solution.add_benchmark(method, {
                                'objective_value': benchmark_solution.objective_value,
                                'execution_time': execution_time,
                                'feasible': benchmark_solution.is_feasible()
                            })
                            
                            if solution.classical_baseline is None:
                                solution.set_classical_baseline(benchmark_solution.objective_value)
                
            except Exception as e:
                self.logger.warning(f"Benchmarking failed for {method}: {e}")
    
    def _select_best_solution(self, solutions: List[Solution]) -> Solution:
        """Select the best solution from multiple candidates."""
        if not solutions:
            # Return empty solution if no solutions found
            return Solution("unknown", SolverType.HYBRID, SolutionStatus.ERROR)
        
        if len(solutions) == 1:
            return solutions[0]
        
        # Rank solutions using the comparator
        ranked_solutions = SolutionComparator.rank_solutions(solutions)
        best_solution = ranked_solutions[0]
        
        # Add alternative solutions
        for alt_solution in ranked_solutions[1:]:
            best_solution.add_alternative_solution(alt_solution)
        
        return best_solution
    
    def _post_process_solution(self, problem: OptimizationProblem, solution: Solution) -> Solution:
        """Post-process the solution for refinement and validation."""
        # Validate constraints
        self._validate_solution_constraints(problem, solution)
        
        # Calculate final metrics
        solution.calculate_feasibility_score()
        solution.finalize_timing()
        
        # Apply any problem-specific post-processing
        if hasattr(problem, 'post_process_solution'):
            solution = problem.post_process_solution(solution)
        
        return solution
    
    def _validate_solution_constraints(self, problem: OptimizationProblem, solution: Solution) -> None:
        """Validate solution against problem constraints."""
        for constraint in problem.constraints:
            try:
                is_satisfied = constraint.evaluate(solution.variables)
                if not is_satisfied:
                    # Calculate violation amount (simplified)
                    violation_amount = 1.0  # This would be calculated based on constraint type
                    solution.add_constraint_violation(
                        constraint.name,
                        violation_amount,
                        constraint.type
                    )
            except Exception as e:
                self.logger.warning(f"Failed to evaluate constraint {constraint.name}: {e}")
    
    def solve_batch(
        self, 
        problems: List[OptimizationProblem],
        **kwargs
    ) -> List[Solution]:
        """Solve multiple problems in batch."""
        self.logger.info(f"Starting batch optimization for {len(problems)} problems")
        
        solutions = []
        for i, problem in enumerate(problems):
            self.logger.info(f"Solving problem {i+1}/{len(problems)}: {problem.name}")
            solution = self.solve(problem, **kwargs)
            solutions.append(solution)
        
        return solutions
    
    def get_solution_history(self) -> List[Solution]:
        """Get the history of all solved problems."""
        return self.solution_history.copy()
    
    def get_cached_solution(self, problem_name: str) -> Optional[Solution]:
        """Get a cached solution by problem name."""
        return self.solution_cache.get(problem_name)
    
    def clear_cache(self) -> None:
        """Clear solution cache."""
        self.solution_cache.clear()
        self.logger.info("Solution cache cleared")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get framework performance metrics."""
        return {
            'total_problems_solved': len(self.solution_history),
            'cached_solutions': len(self.solution_cache),
            'average_solve_time': np.mean([
                s.metadata.execution_time for s in self.solution_history 
                if s.metadata.execution_time > 0
            ]) if self.solution_history else 0,
            'quantum_advantage_rate': sum([
                1 for s in self.solution_history if s.quantum_advantage
            ]) / len(self.solution_history) if self.solution_history else 0,
            'success_rate': sum([
                1 for s in self.solution_history 
                if s.status in [SolutionStatus.OPTIMAL, SolutionStatus.FEASIBLE]
            ]) / len(self.solution_history) if self.solution_history else 0,
            'benchmark_results': self.benchmark_results
        }
    
    def __str__(self) -> str:
        """String representation of the framework."""
        return f"QOptimFramework(problems_solved={len(self.solution_history)})"
    
    def __repr__(self) -> str:
        """Detailed representation of the framework."""
        return self.__str__()

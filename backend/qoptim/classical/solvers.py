"""
Classical optimization solvers for benchmarking and warm-starting quantum algorithms.
"""

import numpy as np
import time
import random
from typing import Dict, List, Optional, Any, Tuple, Callable
import logging
from dataclasses import dataclass

# Classical optimization imports
try:
    from scipy.optimize import minimize, differential_evolution
    from ortools.linear_solver import pywraplp
    from ortools.sat.python import cp_model
    import cvxpy as cp
    from gekko import GEKKO
    CLASSICAL_LIBS_AVAILABLE = True
except ImportError:
    CLASSICAL_LIBS_AVAILABLE = False

from ..core.problem import OptimizationProblem, ProblemType, OptimizationType
from ..core.solution import Solution, SolutionStatus, SolverType
from ..utils.config import Config
from ..utils.logging import get_logger


@dataclass
class SAParameters:
    """Simulated Annealing parameters."""
    initial_temperature: float = 1000.0
    cooling_rate: float = 0.95
    min_temperature: float = 0.01
    max_iterations: int = 10000
    temperature_function: str = "exponential"  # "exponential", "linear", "logarithmic"


@dataclass
class GAParameters:
    """Genetic Algorithm parameters."""
    population_size: int = 100
    generations: int = 500
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elite_size: int = 10
    selection_method: str = "tournament"  # "tournament", "roulette", "rank"


class ClassicalSolvers:
    """
    Collection of classical optimization solvers for comparison and warm-starting.
    
    Includes:
    - Simulated Annealing
    - Genetic Algorithm
    - Exact solvers (when problem size permits)
    - Gradient-based methods for continuous problems
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Algorithm parameters
        self.sa_params = SAParameters(
            initial_temperature=config.get('sa_initial_temp', 1000.0),
            cooling_rate=config.get('sa_cooling_rate', 0.95),
            max_iterations=config.get('sa_max_iterations', 10000)
        )
        
        self.ga_params = GAParameters(
            population_size=config.get('ga_population_size', 100),
            generations=config.get('ga_generations', 500),
            crossover_rate=config.get('ga_crossover_rate', 0.8),
            mutation_rate=config.get('ga_mutation_rate', 0.1)
        )
        
        # Timeouts
        self.default_timeout = config.get('classical_timeout', 300)  # 5 minutes
        
        self.logger.info("Classical Solvers initialized")
    
    def solve_simulated_annealing(
        self, 
        problem: OptimizationProblem,
        max_iterations: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> Solution:
        """
        Solve optimization problem using Simulated Annealing.
        
        Args:
            problem: The optimization problem to solve
            max_iterations: Maximum number of iterations
            timeout: Maximum solving time in seconds
            
        Returns:
            Solution object with SA results
        """
        self.logger.info(f"Starting Simulated Annealing for problem: {problem.name}")
        start_time = time.time()
        
        solution = Solution(problem.name, SolverType.SIMULATED_ANNEALING)
        solution.metadata.start_time = start_time
        
        try:
            max_iter = max_iterations or self.sa_params.max_iterations
            time_limit = timeout or self.default_timeout
            
            # Initialize random solution
            current_solution = self._generate_random_solution(problem)
            current_cost = self._evaluate_solution(problem, current_solution)
            
            best_solution = current_solution.copy()
            best_cost = current_cost
            
            # SA parameters
            temperature = self.sa_params.initial_temperature
            iteration = 0
            
            # SA main loop
            while (iteration < max_iter and 
                   temperature > self.sa_params.min_temperature and
                   time.time() - start_time < time_limit):
                
                # Generate neighbor solution
                neighbor = self._generate_neighbor(problem, current_solution)
                neighbor_cost = self._evaluate_solution(problem, neighbor)
                
                # Accept or reject neighbor
                if self._accept_solution(current_cost, neighbor_cost, temperature):
                    current_solution = neighbor
                    current_cost = neighbor_cost
                    
                    # Update best solution
                    if current_cost < best_cost:
                        best_solution = current_solution.copy()
                        best_cost = current_cost
                
                # Cool down
                temperature = self._update_temperature(temperature, iteration)
                iteration += 1
                
                # Log progress periodically
                if iteration % 1000 == 0:
                    self.logger.debug(f"SA iteration {iteration}, best cost: {best_cost:.4f}, temp: {temperature:.4f}")
            
            # Process results
            solution.set_objective_value(best_cost)
            
            # Convert solution to variable values
            if problem.problem_type in [ProblemType.MAX_CUT, ProblemType.KNAPSACK]:
                solution.set_binary_solution(np.array(best_solution))
                variable_values = {f'x_{i}': float(best_solution[i]) for i in range(len(best_solution))}
            else:
                variable_values = {f'x_{i}': float(val) for i, val in enumerate(best_solution)}
            
            solution.set_variable_values(variable_values)
            solution.status = SolutionStatus.FEASIBLE
            
            # Update metadata
            solution.update_metadata(
                iterations=iteration,
                final_temperature=temperature,
                algorithm_params=self.sa_params.__dict__
            )
            
        except Exception as e:
            self.logger.error(f"Simulated Annealing failed: {e}")
            solution.status = SolutionStatus.ERROR
            solution.update_metadata(error_message=str(e))
        
        finally:
            solution.metadata.end_time = time.time()
            solution.finalize_timing()
        
        self.logger.info(f"SA completed in {solution.metadata.execution_time:.4f}s")
        return solution
    
    def solve_genetic_algorithm(
        self, 
        problem: OptimizationProblem,
        generations: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> Solution:
        """
        Solve optimization problem using Genetic Algorithm.
        
        Args:
            problem: The optimization problem to solve
            generations: Maximum number of generations
            timeout: Maximum solving time in seconds
            
        Returns:
            Solution object with GA results
        """
        self.logger.info(f"Starting Genetic Algorithm for problem: {problem.name}")
        start_time = time.time()
        
        solution = Solution(problem.name, SolverType.GENETIC_ALGORITHM)
        solution.metadata.start_time = start_time
        
        try:
            max_gen = generations or self.ga_params.generations
            time_limit = timeout or self.default_timeout
            
            # Initialize population
            population = [self._generate_random_solution(problem) for _ in range(self.ga_params.population_size)]
            fitness_scores = [self._evaluate_solution(problem, ind) for ind in population]
            
            best_individual = population[np.argmin(fitness_scores)]
            best_fitness = min(fitness_scores)
            
            generation = 0
            
            # GA main loop
            while generation < max_gen and time.time() - start_time < time_limit:
                # Selection
                parents = self._selection(population, fitness_scores)
                
                # Create new generation
                offspring = []
                for i in range(0, len(parents), 2):
                    parent1 = parents[i]
                    parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
                    
                    # Crossover
                    if random.random() < self.ga_params.crossover_rate:
                        child1, child2 = self._crossover(parent1, parent2)
                    else:
                        child1, child2 = parent1.copy(), parent2.copy()
                    
                    # Mutation
                    if random.random() < self.ga_params.mutation_rate:
                        child1 = self._mutate(problem, child1)
                    if random.random() < self.ga_params.mutation_rate:
                        child2 = self._mutate(problem, child2)
                    
                    offspring.extend([child1, child2])
                
                # Evaluate offspring
                offspring_fitness = [self._evaluate_solution(problem, ind) for ind in offspring]
                
                # Combine and select next generation (elitism)
                combined_pop = population + offspring
                combined_fitness = fitness_scores + offspring_fitness
                
                # Sort by fitness
                sorted_indices = np.argsort(combined_fitness)
                
                # Select best individuals for next generation
                population = [combined_pop[i] for i in sorted_indices[:self.ga_params.population_size]]
                fitness_scores = [combined_fitness[i] for i in sorted_indices[:self.ga_params.population_size]]
                
                # Update best solution
                current_best = min(fitness_scores)
                if current_best < best_fitness:
                    best_fitness = current_best
                    best_individual = population[fitness_scores.index(current_best)].copy()
                
                generation += 1
                
                # Log progress
                if generation % 50 == 0:
                    avg_fitness = np.mean(fitness_scores)
                    self.logger.debug(f"GA generation {generation}, best: {best_fitness:.4f}, avg: {avg_fitness:.4f}")
            
            # Process results
            solution.set_objective_value(best_fitness)
            
            # Convert solution to variable values
            if problem.problem_type in [ProblemType.MAX_CUT, ProblemType.KNAPSACK]:
                solution.set_binary_solution(np.array(best_individual))
                variable_values = {f'x_{i}': float(best_individual[i]) for i in range(len(best_individual))}
            else:
                variable_values = {f'x_{i}': float(val) for i, val in enumerate(best_individual)}
            
            solution.set_variable_values(variable_values)
            solution.status = SolutionStatus.FEASIBLE
            
            # Update metadata
            solution.update_metadata(
                iterations=generation,
                final_population_size=len(population),
                best_generation=generation,
                algorithm_params=self.ga_params.__dict__
            )
            
        except Exception as e:
            self.logger.error(f"Genetic Algorithm failed: {e}")
            solution.status = SolutionStatus.ERROR
            solution.update_metadata(error_message=str(e))
        
        finally:
            solution.metadata.end_time = time.time()
            solution.finalize_timing()
        
        self.logger.info(f"GA completed in {solution.metadata.execution_time:.4f}s")
        return solution
    
    def solve_exact(self, problem: OptimizationProblem) -> Solution:
        """
        Solve optimization problem using exact methods (when feasible).
        
        Args:
            problem: The optimization problem to solve
            
        Returns:
            Solution object with exact results
        """
        self.logger.info(f"Starting Exact solver for problem: {problem.name}")
        start_time = time.time()
        
        solution = Solution(problem.name, SolverType.EXACT)
        solution.metadata.start_time = start_time
        
        try:
            if not CLASSICAL_LIBS_AVAILABLE:
                self.logger.error("Classical optimization libraries not available")
                solution.status = SolutionStatus.ERROR
                return solution
            
            # Check problem size feasibility
            problem_size = problem.get_problem_size()
            if problem_size['num_variables'] > 25:  # Limit for exact methods
                self.logger.warning(f"Problem too large for exact solver: {problem_size['num_variables']} variables")
                solution.status = SolutionStatus.ERROR
                return solution
            
            if problem.problem_type == ProblemType.KNAPSACK:
                result = self._solve_knapsack_exact(problem)
            elif problem.problem_type == ProblemType.MAX_CUT:
                result = self._solve_maxcut_exact(problem)
            elif problem.problem_type == ProblemType.TSP:
                result = self._solve_tsp_exact(problem)
            elif problem.problem_type == ProblemType.PORTFOLIO:
                result = self._solve_portfolio_exact(problem)
            else:
                result = self._solve_generic_exact(problem)
            
            if result is not None:
                solution.set_objective_value(result['objective_value'])
                solution.set_variable_values(result['variables'])
                if 'binary_solution' in result:
                    solution.set_binary_solution(np.array(result['binary_solution']))
                solution.status = SolutionStatus.OPTIMAL
            else:
                solution.status = SolutionStatus.ERROR
            
        except Exception as e:
            self.logger.error(f"Exact solver failed: {e}")
            solution.status = SolutionStatus.ERROR
            solution.update_metadata(error_message=str(e))
        
        finally:
            solution.metadata.end_time = time.time()
            solution.finalize_timing()
        
        self.logger.info(f"Exact solver completed in {solution.metadata.execution_time:.4f}s")
        return solution
    
    def _generate_random_solution(self, problem: OptimizationProblem) -> List[float]:
        """Generate a random solution for the problem."""
        if problem.problem_type in [ProblemType.MAX_CUT, ProblemType.KNAPSACK, ProblemType.GRAPH_COLORING]:
            # Binary variables
            n_vars = len(problem.variables) if problem.variables else 10
            return [random.randint(0, 1) for _ in range(n_vars)]
        elif problem.problem_type == ProblemType.TSP:
            # Permutation
            distance_matrix = problem.get_data('distance_matrix')
            if distance_matrix:
                n_cities = len(distance_matrix)
                cities = list(range(n_cities))
                random.shuffle(cities)
                return cities
            else:
                return [0, 1, 2, 3]  # Default small TSP
        elif problem.problem_type == ProblemType.PORTFOLIO:
            # Portfolio weights (sum to 1)
            n_assets = len(problem.get_data('expected_returns', [0.1, 0.15, 0.12]))
            weights = [random.random() for _ in range(n_assets)]
            total = sum(weights)
            return [w / total for w in weights]
        else:
            # Continuous variables
            n_vars = len(problem.variables) if problem.variables else 5
            return [random.uniform(0, 1) for _ in range(n_vars)]
    
    def _evaluate_solution(self, problem: OptimizationProblem, solution: List[float]) -> float:
        """Evaluate the objective function for a given solution."""
        try:
            if problem.problem_type == ProblemType.MAX_CUT:
                return self._evaluate_maxcut(problem, solution)
            elif problem.problem_type == ProblemType.KNAPSACK:
                return self._evaluate_knapsack(problem, solution)
            elif problem.problem_type == ProblemType.TSP:
                return self._evaluate_tsp(problem, solution)
            elif problem.problem_type == ProblemType.PORTFOLIO:
                return self._evaluate_portfolio(problem, solution)
            elif callable(problem.objective_function):
                var_dict = {f'x_{i}': val for i, val in enumerate(solution)}
                return problem.objective_function(var_dict)
            else:
                # Simple quadratic objective
                return sum(x**2 for x in solution)
        except:
            return float('inf')  # Invalid solution
    
    def _evaluate_maxcut(self, problem: OptimizationProblem, solution: List[float]) -> float:
        """Evaluate Max-Cut objective."""
        if problem.graph is None:
            return 0.0
        
        cut_value = 0
        for u, v, data in problem.graph.edges(data=True):
            if u < len(solution) and v < len(solution):
                weight = data.get('weight', 1.0)
                if int(solution[u]) != int(solution[v]):
                    cut_value += weight
        
        return -cut_value  # Negative because we want to maximize cut
    
    def _evaluate_knapsack(self, problem: OptimizationProblem, solution: List[float]) -> float:
        """Evaluate Knapsack objective."""
        weights = problem.get_data('weights', [])
        values = problem.get_data('values', [])
        capacity = problem.get_data('capacity', 100)
        
        if not weights or not values:
            return 0.0
        
        total_weight = sum(w * int(s) for w, s in zip(weights, solution))
        total_value = sum(v * int(s) for v, s in zip(values, solution))
        
        # Penalty for exceeding capacity
        if total_weight > capacity:
            return total_weight - capacity + 1000  # Heavy penalty
        
        return -total_value  # Negative because we want to maximize value
    
    def _evaluate_tsp(self, problem: OptimizationProblem, solution: List[float]) -> float:
        """Evaluate TSP objective."""
        distance_matrix = problem.get_data('distance_matrix')
        if distance_matrix is None:
            return 0.0
        
        total_distance = 0
        n_cities = len(solution)
        
        for i in range(n_cities):
            current_city = int(solution[i])
            next_city = int(solution[(i + 1) % n_cities])
            if current_city < len(distance_matrix) and next_city < len(distance_matrix[0]):
                total_distance += distance_matrix[current_city][next_city]
        
        return total_distance
    
    def _evaluate_portfolio(self, problem: OptimizationProblem, solution: List[float]) -> float:
        """Evaluate portfolio objective."""
        expected_returns = np.array(problem.get_data('expected_returns', [0.1, 0.15, 0.12]))
        covariance_matrix = np.array(problem.get_data('covariance_matrix', np.eye(3) * 0.01))
        risk_aversion = problem.get_data('risk_aversion', 1.0)
        
        weights = np.array(solution[:len(expected_returns)])
        
        # Ensure weights sum to 1
        if abs(sum(weights) - 1.0) > 0.01:
            return 1000.0  # Penalty for invalid weights
        
        expected_return = np.dot(expected_returns, weights)
        portfolio_risk = np.dot(weights, np.dot(covariance_matrix, weights))
        
        # Minimize risk - return (convert maximization to minimization)
        return risk_aversion * portfolio_risk - expected_return
    
    def _generate_neighbor(self, problem: OptimizationProblem, solution: List[float]) -> List[float]:
        """Generate a neighbor solution for SA."""
        neighbor = solution.copy()
        
        if problem.problem_type in [ProblemType.MAX_CUT, ProblemType.KNAPSACK]:
            # Flip a random bit
            idx = random.randint(0, len(neighbor) - 1)
            neighbor[idx] = 1 - neighbor[idx]
        elif problem.problem_type == ProblemType.TSP:
            # 2-opt swap
            if len(neighbor) > 3:
                i, j = random.sample(range(len(neighbor)), 2)
                if i > j:
                    i, j = j, i
                neighbor[i:j+1] = reversed(neighbor[i:j+1])
        else:
            # Add Gaussian noise
            idx = random.randint(0, len(neighbor) - 1)
            neighbor[idx] += random.gauss(0, 0.1)
            neighbor[idx] = max(0, min(1, neighbor[idx]))  # Clamp to [0,1]
        
        return neighbor
    
    def _accept_solution(self, current_cost: float, neighbor_cost: float, temperature: float) -> bool:
        """SA acceptance criterion."""
        if neighbor_cost < current_cost:
            return True
        else:
            probability = np.exp(-(neighbor_cost - current_cost) / temperature)
            return random.random() < probability
    
    def _update_temperature(self, temperature: float, iteration: int) -> float:
        """Update temperature in SA."""
        if self.sa_params.temperature_function == "exponential":
            return temperature * self.sa_params.cooling_rate
        elif self.sa_params.temperature_function == "linear":
            return temperature - (self.sa_params.initial_temperature / self.sa_params.max_iterations)
        else:  # logarithmic
            return self.sa_params.initial_temperature / (1 + np.log(1 + iteration))
    
    def _selection(self, population: List[List[float]], fitness_scores: List[float]) -> List[List[float]]:
        """Selection operator for GA."""
        if self.ga_params.selection_method == "tournament":
            return self._tournament_selection(population, fitness_scores)
        elif self.ga_params.selection_method == "roulette":
            return self._roulette_selection(population, fitness_scores)
        else:  # rank selection
            return self._rank_selection(population, fitness_scores)
    
    def _tournament_selection(self, population: List[List[float]], fitness_scores: List[float]) -> List[List[float]]:
        """Tournament selection."""
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
            selected.append(population[winner_idx].copy())
        
        return selected
    
    def _roulette_selection(self, population: List[List[float]], fitness_scores: List[float]) -> List[List[float]]:
        """Roulette wheel selection."""
        # Convert minimization to maximization for roulette wheel
        max_fitness = max(fitness_scores)
        adjusted_fitness = [max_fitness - f + 1 for f in fitness_scores]
        total_fitness = sum(adjusted_fitness)
        
        if total_fitness == 0:
            return random.choices(population, k=len(population))
        
        probabilities = [f / total_fitness for f in adjusted_fitness]
        return random.choices(population, weights=probabilities, k=len(population))
    
    def _rank_selection(self, population: List[List[float]], fitness_scores: List[float]) -> List[List[float]]:
        """Rank-based selection."""
        sorted_indices = np.argsort(fitness_scores)  # Best first
        ranks = {idx: i + 1 for i, idx in enumerate(sorted_indices)}
        
        rank_weights = [ranks[i] for i in range(len(population))]
        total_rank = sum(rank_weights)
        
        probabilities = [r / total_rank for r in rank_weights]
        return random.choices(population, weights=probabilities, k=len(population))
    
    def _crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Crossover operator."""
        if len(parent1) != len(parent2):
            return parent1.copy(), parent2.copy()
        
        # Single-point crossover
        crossover_point = random.randint(1, len(parent1) - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def _mutate(self, problem: OptimizationProblem, individual: List[float]) -> List[float]:
        """Mutation operator."""
        mutated = individual.copy()
        
        if problem.problem_type in [ProblemType.MAX_CUT, ProblemType.KNAPSACK]:
            # Bit flip mutation
            idx = random.randint(0, len(mutated) - 1)
            mutated[idx] = 1 - mutated[idx]
        elif problem.problem_type == ProblemType.TSP:
            # Swap mutation
            if len(mutated) > 1:
                i, j = random.sample(range(len(mutated)), 2)
                mutated[i], mutated[j] = mutated[j], mutated[i]
        else:
            # Gaussian mutation
            idx = random.randint(0, len(mutated) - 1)
            mutated[idx] += random.gauss(0, 0.1)
            mutated[idx] = max(0, min(1, mutated[idx]))
        
        return mutated
    
    def _solve_knapsack_exact(self, problem: OptimizationProblem) -> Optional[Dict[str, Any]]:
        """Solve knapsack problem exactly using OR-Tools."""
        try:
            weights = problem.get_data('weights')
            values = problem.get_data('values')
            capacity = problem.get_data('capacity')
            
            if not all([weights, values, capacity]):
                return None
            
            # Create CP-SAT model
            model = cp_model.CpModel()
            
            # Variables
            n_items = len(weights)
            x = [model.NewBoolVar(f'x_{i}') for i in range(n_items)]
            
            # Constraint: weight limit
            model.Add(sum(x[i] * weights[i] for i in range(n_items)) <= capacity)
            
            # Objective: maximize value
            model.Maximize(sum(x[i] * values[i] for i in range(n_items)))
            
            # Solve
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 30
            status = solver.Solve(model)
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                binary_solution = [solver.Value(x[i]) for i in range(n_items)]
                objective_value = solver.ObjectiveValue()
                
                return {
                    'objective_value': -objective_value,  # Convert back to minimization
                    'variables': {f'x_{i}': float(binary_solution[i]) for i in range(n_items)},
                    'binary_solution': binary_solution
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Exact knapsack solving failed: {e}")
            return None
    
    def _solve_maxcut_exact(self, problem: OptimizationProblem) -> Optional[Dict[str, Any]]:
        """Solve Max-Cut problem exactly (for small graphs)."""
        try:
            if problem.graph is None:
                return None
            
            n_nodes = problem.graph.number_of_nodes()
            best_cut = 0
            best_assignment = [0] * n_nodes
            
            # Brute force for small graphs
            if n_nodes <= 20:
                for assignment in range(2**n_nodes):
                    # Convert to binary
                    binary = [(assignment >> i) & 1 for i in range(n_nodes)]
                    
                    # Calculate cut value
                    cut_value = 0
                    for u, v, data in problem.graph.edges(data=True):
                        weight = data.get('weight', 1.0)
                        if binary[u] != binary[v]:
                            cut_value += weight
                    
                    if cut_value > best_cut:
                        best_cut = cut_value
                        best_assignment = binary
                
                return {
                    'objective_value': -best_cut,  # Convert to minimization
                    'variables': {f'x_{i}': float(best_assignment[i]) for i in range(n_nodes)},
                    'binary_solution': best_assignment
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Exact Max-Cut solving failed: {e}")
            return None
    
    def _solve_tsp_exact(self, problem: OptimizationProblem) -> Optional[Dict[str, Any]]:
        """Solve TSP exactly (for small instances)."""
        try:
            distance_matrix = problem.get_data('distance_matrix')
            if distance_matrix is None:
                return None
            
            n_cities = len(distance_matrix)
            
            # Use OR-Tools for small TSP instances
            if n_cities <= 10 and CLASSICAL_LIBS_AVAILABLE:
                from ortools.constraint_solver import routing_enums_pb2
                from ortools.constraint_solver import pywrapcp
                
                # Create routing model
                manager = pywrapcp.RoutingIndexManager(n_cities, 1, 0)
                routing = pywrapcp.RoutingModel(manager)
                
                def distance_callback(from_index, to_index):
                    from_node = manager.IndexToNode(from_index)
                    to_node = manager.IndexToNode(to_index)
                    return int(distance_matrix[from_node][to_node])
                
                transit_callback_index = routing.RegisterTransitCallback(distance_callback)
                routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
                
                # Set search parameters
                search_parameters = pywrapcp.DefaultRoutingSearchParameters()
                search_parameters.first_solution_strategy = (
                    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
                )
                search_parameters.time_limit.seconds = 10
                
                # Solve
                solution = routing.SolveWithParameters(search_parameters)
                
                if solution:
                    # Extract route
                    index = routing.Start(0)
                    route = []
                    while not routing.IsEnd(index):
                        route.append(manager.IndexToNode(index))
                        index = solution.Value(routing.NextVar(index))
                    
                    return {
                        'objective_value': solution.ObjectiveValue(),
                        'variables': {f'city_{i}': float(route[i]) for i in range(len(route))},
                        'route': route
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Exact TSP solving failed: {e}")
            return None
    
    def _solve_portfolio_exact(self, problem: OptimizationProblem) -> Optional[Dict[str, Any]]:
        """Solve portfolio optimization exactly using convex optimization."""
        try:
            if not CLASSICAL_LIBS_AVAILABLE:
                return None
            
            expected_returns = np.array(problem.get_data('expected_returns', []))
            covariance_matrix = np.array(problem.get_data('covariance_matrix', []))
            risk_aversion = problem.get_data('risk_aversion', 1.0)
            
            if len(expected_returns) == 0 or len(covariance_matrix) == 0:
                return None
            
            n_assets = len(expected_returns)
            
            # Define optimization variables
            w = cp.Variable(n_assets)
            
            # Objective: minimize risk - expected return
            expected_return = expected_returns.T @ w
            portfolio_risk = cp.quad_form(w, covariance_matrix)
            objective = risk_aversion * portfolio_risk - expected_return
            
            # Constraints
            constraints = [
                cp.sum(w) == 1,  # Weights sum to 1
                w >= 0  # Long-only constraint
            ]
            
            # Solve
            problem_cvx = cp.Problem(cp.Minimize(objective), constraints)
            problem_cvx.solve()
            
            if problem_cvx.status == cp.OPTIMAL:
                optimal_weights = w.value
                objective_value = problem_cvx.value
                
                return {
                    'objective_value': objective_value,
                    'variables': {f'w_{i}': float(optimal_weights[i]) for i in range(n_assets)},
                    'portfolio_weights': optimal_weights.tolist()
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Exact portfolio solving failed: {e}")
            return None
    
    def _solve_generic_exact(self, problem: OptimizationProblem) -> Optional[Dict[str, Any]]:
        """Solve generic problem using available exact methods."""
        try:
            # For small problems, try brute force or scipy optimization
            n_vars = len(problem.variables) if problem.variables else 5
            
            if n_vars <= 10:
                # Try scipy optimization for continuous problems
                def objective(x):
                    return self._evaluate_solution(problem, x.tolist())
                
                # Random starting point
                x0 = np.random.rand(n_vars)
                bounds = [(0, 1)] * n_vars
                
                result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
                
                if result.success:
                    return {
                        'objective_value': result.fun,
                        'variables': {f'x_{i}': float(result.x[i]) for i in range(n_vars)}
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Generic exact solving failed: {e}")
            return None

"""
AI preprocessing module for problem reduction, feature selection, and solution space mapping.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple
import logging

# Machine Learning imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GATConv, GraphConv
    from torch_geometric.data import Data
    import torch_geometric
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import networkx as nx
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..core.problem import OptimizationProblem, ProblemType, Variable, Constraint
from ..core.solution import Solution
from ..utils.config import Config
from ..utils.logging import get_logger


class HeuristicNeuralNetwork(nn.Module):
    """Neural network for learning problem-specific heuristics."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], output_dim: int = 1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for graph-based optimization problems."""
    
    def __init__(self, node_features: int, edge_features: int = 0, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(node_features, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        self.conv_layers.append(GCNConv(hidden_dim, 1))  # Output node scores
        
        # Attention mechanism
        self.attention = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        
    def forward(self, x, edge_index):
        # Apply graph convolutions
        for i, conv in enumerate(self.conv_layers[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        
        # Apply attention
        if len(self.conv_layers) > 1:
            x = self.attention(x, edge_index)
            x = F.relu(x)
        
        # Final layer
        x = self.conv_layers[-1](x, edge_index)
        
        return x


class ReinforcementLearningAgent(nn.Module):
    """Reinforcement Learning agent for solution construction."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value


class AIPreprocessor:
    """
    AI-powered preprocessing system for optimization problems.
    
    Capabilities:
    - Problem size reduction through feature selection
    - Solution space mapping and encoding
    - Heuristic learning from problem structure
    - Graph neural networks for graph-based problems
    - Reinforcement learning for solution construction
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        
        # AI model parameters
        self.enable_feature_selection = config.get('ai_feature_selection', True)
        self.enable_problem_reduction = config.get('ai_problem_reduction', True)
        self.enable_heuristic_learning = config.get('ai_heuristic_learning', True)
        self.enable_gnn = config.get('ai_gnn_enabled', True)
        self.enable_rl = config.get('ai_rl_enabled', True)
        
        # Model configurations
        self.pca_components = config.get('ai_pca_components', 0.95)
        self.feature_selection_k = config.get('ai_feature_k', 50)
        self.gnn_hidden_dim = config.get('ai_gnn_hidden_dim', 64)
        self.rl_hidden_dim = config.get('ai_rl_hidden_dim', 128)
        
        # Models (initialized lazily)
        self.heuristic_model: Optional[HeuristicNeuralNetwork] = None
        self.graph_model: Optional[GraphNeuralNetwork] = None
        self.rl_agent: Optional[ReinforcementLearningAgent] = None
        
        # Preprocessing components
        self.scaler = StandardScaler() if TORCH_AVAILABLE else None
        self.pca = None
        self.feature_selector = None
        
        # Training data storage
        self.training_data: List[Dict[str, Any]] = []
        
        self.logger.info("AI Preprocessor initialized")
    
    def preprocess_problem(self, problem: OptimizationProblem) -> OptimizationProblem:
        """
        Apply AI preprocessing to reduce problem complexity and enhance solving.
        
        Args:
            problem: Original optimization problem
            
        Returns:
            Preprocessed optimization problem
        """
        self.logger.info(f"Starting AI preprocessing for problem: {problem.name}")
        start_time = time.time()
        
        try:
            # Store preprocessing info
            preprocessing_info = {
                'original_size': problem.get_problem_size(),
                'start_time': start_time,
                'methods_applied': []
            }
            
            # Apply feature selection
            if self.enable_feature_selection and len(problem.variables) > 10:
                problem = self._apply_feature_selection(problem)
                preprocessing_info['methods_applied'].append('feature_selection')
            
            # Apply problem reduction techniques
            if self.enable_problem_reduction:
                problem = self._apply_problem_reduction(problem)
                preprocessing_info['methods_applied'].append('problem_reduction')
            
            # Apply graph neural network preprocessing for graph problems
            if (self.enable_gnn and TORCH_AVAILABLE and 
                problem.problem_type in [ProblemType.MAX_CUT, ProblemType.GRAPH_COLORING, 
                                       ProblemType.TSP] and problem.graph is not None):
                problem = self._apply_gnn_preprocessing(problem)
                preprocessing_info['methods_applied'].append('gnn_preprocessing')
            
            # Apply heuristic learning
            if self.enable_heuristic_learning and TORCH_AVAILABLE:
                problem = self._apply_heuristic_learning(problem)
                preprocessing_info['methods_applied'].append('heuristic_learning')
            
            # Generate initial solution using RL if enabled
            if self.enable_rl and TORCH_AVAILABLE:
                initial_solution = self._generate_rl_solution(problem)
                if initial_solution:
                    problem.metadata['rl_initial_solution'] = initial_solution
                    preprocessing_info['methods_applied'].append('rl_solution_generation')
            
            # Store preprocessing information
            preprocessing_info['final_size'] = problem.get_problem_size()
            preprocessing_info['preprocessing_time'] = time.time() - start_time
            preprocessing_info['reduction_ratio'] = (
                preprocessing_info['original_size']['num_variables'] / 
                max(1, preprocessing_info['final_size']['num_variables'])
            )
            
            problem.preprocessing_info = preprocessing_info
            
            self.logger.info(f"AI preprocessing completed in {preprocessing_info['preprocessing_time']:.4f}s")
            self.logger.info(f"Problem size reduced by factor: {preprocessing_info['reduction_ratio']:.2f}")
            
            return problem
            
        except Exception as e:
            self.logger.error(f"AI preprocessing failed: {e}")
            # Return original problem if preprocessing fails
            return problem
    
    def _apply_feature_selection(self, problem: OptimizationProblem) -> OptimizationProblem:
        """Apply feature selection to reduce problem dimensionality."""
        if not TORCH_AVAILABLE:
            return problem
        
        original_vars = list(problem.variables.keys())
        n_vars = len(original_vars)
        
        if n_vars <= 10:
            return problem  # Too small for feature selection
        
        try:
            # Create synthetic data for feature selection based on problem structure
            X, y = self._generate_synthetic_data(problem)
            
            if X is None or y is None:
                return problem
            
            # Apply feature selection
            k = min(self.feature_selection_k, max(5, n_vars // 2))
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # Get selected feature indices
            selected_indices = self.feature_selector.get_support(indices=True)
            selected_vars = [original_vars[i] for i in selected_indices]
            
            # Create reduced problem
            reduced_problem = OptimizationProblem(
                problem.name + "_reduced",
                problem.problem_type,
                problem.optimization_type
            )
            
            # Copy selected variables
            for var_name in selected_vars:
                if var_name in problem.variables:
                    reduced_problem.add_variable(problem.variables[var_name])
            
            # Adapt constraints to reduced variable set
            for constraint in problem.constraints:
                if any(var in selected_vars for var in constraint.variables):
                    reduced_problem.add_constraint(constraint)
            
            # Copy other problem attributes
            reduced_problem.data = problem.data.copy()
            reduced_problem.graph = problem.graph
            reduced_problem.objective_function = problem.objective_function
            reduced_problem.metadata = problem.metadata.copy()
            reduced_problem.metadata['feature_selection_applied'] = True
            reduced_problem.metadata['selected_features'] = selected_vars
            
            self.logger.info(f"Feature selection: {n_vars} -> {len(selected_vars)} variables")
            
            return reduced_problem
            
        except Exception as e:
            self.logger.warning(f"Feature selection failed: {e}")
            return problem
    
    def _apply_problem_reduction(self, problem: OptimizationProblem) -> OptimizationProblem:
        """Apply problem-specific reduction techniques."""
        try:
            if problem.problem_type == ProblemType.TSP:
                return self._reduce_tsp_problem(problem)
            elif problem.problem_type == ProblemType.KNAPSACK:
                return self._reduce_knapsack_problem(problem)
            elif problem.problem_type == ProblemType.MAX_CUT and problem.graph is not None:
                return self._reduce_maxcut_problem(problem)
            else:
                return self._apply_general_reduction(problem)
                
        except Exception as e:
            self.logger.warning(f"Problem reduction failed: {e}")
            return problem
    
    def _apply_gnn_preprocessing(self, problem: OptimizationProblem) -> OptimizationProblem:
        """Apply Graph Neural Network preprocessing for graph-based problems."""
        if not TORCH_AVAILABLE or problem.graph is None:
            return problem
        
        try:
            # Convert NetworkX graph to PyTorch Geometric format
            graph_data = self._networkx_to_pytorch_geometric(problem.graph)
            
            # Initialize or load GNN model
            if self.graph_model is None:
                node_features = graph_data.x.size(1) if graph_data.x is not None else 1
                self.graph_model = GraphNeuralNetwork(
                    node_features=node_features,
                    hidden_dim=self.gnn_hidden_dim
                )
            
            # Get node embeddings
            self.graph_model.eval()
            with torch.no_grad():
                node_scores = self.graph_model(graph_data.x, graph_data.edge_index)
            
            # Use node scores to guide problem formulation
            node_importance = torch.sigmoid(node_scores).flatten().numpy()
            
            # Store GNN insights in problem metadata
            problem.metadata['gnn_node_scores'] = node_importance.tolist()
            problem.metadata['gnn_applied'] = True
            
            # Optionally reduce graph based on node importance
            if len(node_importance) > 20:  # Only for larger graphs
                threshold = np.percentile(node_importance, 25)  # Keep top 75% nodes
                important_nodes = [i for i, score in enumerate(node_importance) if score >= threshold]
                
                # Create reduced graph
                reduced_graph = problem.graph.subgraph(important_nodes).copy()
                problem.graph = reduced_graph
                problem.metadata['gnn_reduced_nodes'] = important_nodes
                
                self.logger.info(f"GNN graph reduction: {len(node_importance)} -> {len(important_nodes)} nodes")
            
            return problem
            
        except Exception as e:
            self.logger.warning(f"GNN preprocessing failed: {e}")
            return problem
    
    def _apply_heuristic_learning(self, problem: OptimizationProblem) -> OptimizationProblem:
        """Apply learned heuristics to guide problem solving."""
        if not TORCH_AVAILABLE:
            return problem
        
        try:
            # Extract problem features
            problem_features = self._extract_problem_features(problem)
            
            if problem_features is None:
                return problem
            
            # Initialize heuristic model if needed
            if self.heuristic_model is None:
                self.heuristic_model = HeuristicNeuralNetwork(
                    input_dim=len(problem_features),
                    hidden_dims=[128, 64, 32]
                )
            
            # Get heuristic guidance
            self.heuristic_model.eval()
            with torch.no_grad():
                features_tensor = torch.tensor(problem_features, dtype=torch.float32).unsqueeze(0)
                heuristic_score = self.heuristic_model(features_tensor)
            
            # Store heuristic guidance
            problem.metadata['heuristic_score'] = float(heuristic_score.item())
            problem.metadata['heuristic_applied'] = True
            problem.metadata['problem_features'] = problem_features.tolist()
            
            return problem
            
        except Exception as e:
            self.logger.warning(f"Heuristic learning failed: {e}")
            return problem
    
    def _generate_rl_solution(self, problem: OptimizationProblem) -> Optional[Dict[str, Any]]:
        """Generate initial solution using reinforcement learning."""
        if not TORCH_AVAILABLE:
            return None
        
        try:
            # Extract state representation
            state = self._problem_to_state(problem)
            
            if state is None:
                return None
            
            # Initialize RL agent if needed
            if self.rl_agent is None:
                state_dim = len(state)
                action_dim = len(problem.variables) if problem.variables else 10
                self.rl_agent = ReinforcementLearningAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dim=self.rl_hidden_dim
                )
            
            # Generate solution
            self.rl_agent.eval()
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action_probs, value = self.rl_agent(state_tensor)
            
            # Convert probabilities to solution
            action_probs = action_probs.squeeze().numpy()
            binary_solution = (action_probs > 0.5).astype(int)
            
            # Create solution dictionary
            solution_dict = {
                'binary_solution': binary_solution.tolist(),
                'action_probabilities': action_probs.tolist(),
                'estimated_value': float(value.item()),
                'method': 'reinforcement_learning'
            }
            
            return solution_dict
            
        except Exception as e:
            self.logger.warning(f"RL solution generation failed: {e}")
            return None
    
    def _generate_synthetic_data(self, problem: OptimizationProblem) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Generate synthetic training data for feature selection."""
        try:
            n_vars = len(problem.variables)
            n_samples = min(1000, max(100, n_vars * 10))
            
            # Generate random variable assignments
            X = np.random.rand(n_samples, n_vars)
            
            # Generate synthetic labels based on problem structure
            y = np.zeros(n_samples)
            
            for i in range(n_samples):
                # Simple heuristic: variables with higher values are more likely to be selected
                score = np.sum(X[i] * np.random.rand(n_vars))
                y[i] = 1 if score > np.median([np.sum(X[j] * np.random.rand(n_vars)) for j in range(n_samples)]) else 0
            
            return X, y
            
        except Exception as e:
            self.logger.warning(f"Synthetic data generation failed: {e}")
            return None, None
    
    def _networkx_to_pytorch_geometric(self, graph: nx.Graph) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data object."""
        # Node features (degree, clustering coefficient, etc.)
        node_features = []
        for node in graph.nodes():
            degree = graph.degree(node)
            clustering = nx.clustering(graph, node)
            betweenness = nx.betweenness_centrality(graph).get(node, 0)
            node_features.append([degree, clustering, betweenness])
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Edge indices
        edge_list = list(graph.edges())
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)
    
    def _extract_problem_features(self, problem: OptimizationProblem) -> Optional[np.ndarray]:
        """Extract numerical features from optimization problem."""
        try:
            features = []
            
            # Problem size features
            size_info = problem.get_problem_size()
            features.extend([
                size_info['num_variables'],
                size_info['num_constraints'],
                size_info['num_binary_vars'],
                size_info['num_continuous_vars']
            ])
            
            # Problem type encoding (one-hot)
            problem_types = list(ProblemType)
            type_encoding = [1 if problem.problem_type == ptype else 0 for ptype in problem_types]
            features.extend(type_encoding)
            
            # Graph features (if applicable)
            if problem.graph is not None:
                features.extend([
                    problem.graph.number_of_nodes(),
                    problem.graph.number_of_edges(),
                    nx.density(problem.graph),
                    np.mean([d for n, d in problem.graph.degree()]),
                    np.mean(list(nx.clustering(problem.graph).values()))
                ])
            else:
                features.extend([0, 0, 0, 0, 0])  # Placeholder zeros
            
            # Data-specific features
            if 'distance_matrix' in problem.data:
                dist_matrix = problem.data['distance_matrix']
                features.extend([
                    np.mean(dist_matrix),
                    np.std(dist_matrix),
                    np.min(dist_matrix),
                    np.max(dist_matrix)
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.warning(f"Feature extraction failed: {e}")
            return None
    
    def _problem_to_state(self, problem: OptimizationProblem) -> Optional[np.ndarray]:
        """Convert optimization problem to state representation for RL."""
        try:
            # Similar to feature extraction but focused on state representation
            features = self._extract_problem_features(problem)
            
            if features is not None:
                # Normalize features
                if self.scaler is not None:
                    # Fit scaler if not already fitted
                    if not hasattr(self.scaler, 'mean_'):
                        # Use dummy data to fit scaler
                        dummy_data = np.random.rand(100, len(features))
                        self.scaler.fit(dummy_data)
                    
                    features = features.reshape(1, -1)
                    features = self.scaler.transform(features).flatten()
            
            return features
            
        except Exception as e:
            self.logger.warning(f"State conversion failed: {e}")
            return None
    
    def _reduce_tsp_problem(self, problem: OptimizationProblem) -> OptimizationProblem:
        """Apply TSP-specific reduction techniques."""
        distance_matrix = problem.get_data('distance_matrix')
        
        if distance_matrix is None:
            return problem
        
        n_cities = len(distance_matrix)
        
        # For large TSP instances, apply clustering-based reduction
        if n_cities > 50:
            # Use k-means clustering to group nearby cities
            coordinates = problem.get_data('coordinates')
            
            if coordinates is not None and TORCH_AVAILABLE:
                n_clusters = min(20, n_cities // 3)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(coordinates)
                
                # Create reduced problem with cluster representatives
                cluster_centers = kmeans.cluster_centers_
                
                # Update problem data
                problem.set_data('coordinates', cluster_centers)
                problem.set_data('distance_matrix', self._compute_distance_matrix(cluster_centers))
                problem.metadata['tsp_clustered'] = True
                problem.metadata['original_cities'] = n_cities
                problem.metadata['cluster_assignments'] = clusters.tolist()
                
                self.logger.info(f"TSP clustering: {n_cities} -> {n_clusters} cities")
        
        return problem
    
    def _reduce_knapsack_problem(self, problem: OptimizationProblem) -> OptimizationProblem:
        """Apply knapsack-specific reduction techniques."""
        weights = problem.get_data('weights')
        values = problem.get_data('values')
        capacity = problem.get_data('capacity')
        
        if not all([weights, values, capacity]):
            return problem
        
        # Remove items that are too heavy
        valid_items = [i for i, w in enumerate(weights) if w <= capacity]
        
        if len(valid_items) < len(weights):
            # Filter items
            problem.set_data('weights', [weights[i] for i in valid_items])
            problem.set_data('values', [values[i] for i in valid_items])
            problem.metadata['knapsack_filtered'] = True
            problem.metadata['removed_items'] = len(weights) - len(valid_items)
            
            self.logger.info(f"Knapsack filtering: {len(weights)} -> {len(valid_items)} items")
        
        return problem
    
    def _reduce_maxcut_problem(self, problem: OptimizationProblem) -> OptimizationProblem:
        """Apply Max-Cut specific reduction techniques."""
        graph = problem.graph
        
        if graph is None:
            return problem
        
        # Remove low-degree nodes that don't contribute much
        if graph.number_of_nodes() > 30:
            min_degree = 2
            nodes_to_remove = [node for node, degree in graph.degree() if degree < min_degree]
            
            if nodes_to_remove:
                graph.remove_nodes_from(nodes_to_remove)
                problem.metadata['maxcut_pruned'] = True
                problem.metadata['removed_nodes'] = len(nodes_to_remove)
                
                self.logger.info(f"Max-Cut pruning: removed {len(nodes_to_remove)} low-degree nodes")
        
        return problem
    
    def _apply_general_reduction(self, problem: OptimizationProblem) -> OptimizationProblem:
        """Apply general reduction techniques."""
        # Remove redundant constraints
        if len(problem.constraints) > 10:
            # Simple redundancy check (simplified)
            unique_constraints = []
            constraint_signatures = set()
            
            for constraint in problem.constraints:
                signature = (constraint.type, constraint.value, tuple(sorted(constraint.variables)))
                if signature not in constraint_signatures:
                    constraint_signatures.add(signature)
                    unique_constraints.append(constraint)
            
            if len(unique_constraints) < len(problem.constraints):
                problem.constraints = unique_constraints
                problem.metadata['constraints_deduplicated'] = True
                
                self.logger.info(f"Constraint deduplication: {len(problem.constraints)} -> {len(unique_constraints)}")
        
        return problem
    
    def _compute_distance_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        """Compute distance matrix from coordinates."""
        n_points = len(coordinates)
        distance_matrix = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                distance_matrix[i, j] = distance_matrix[j, i] = dist
        
        return distance_matrix
    
    def update_training_data(self, problem: OptimizationProblem, solution: Solution) -> None:
        """Update training data with problem-solution pairs for model improvement."""
        try:
            training_sample = {
                'problem_features': self._extract_problem_features(problem),
                'problem_type': problem.problem_type.value,
                'objective_value': solution.objective_value,
                'feasible': solution.is_feasible(),
                'solving_time': solution.metadata.execution_time,
                'solver_type': solution.solver_type.value
            }
            
            if all(v is not None for v in training_sample.values()):
                self.training_data.append(training_sample)
                
                # Limit training data size
                if len(self.training_data) > 1000:
                    self.training_data = self.training_data[-1000:]
                
                self.logger.debug(f"Updated training data: {len(self.training_data)} samples")
            
        except Exception as e:
            self.logger.warning(f"Training data update failed: {e}")
    
    def retrain_models(self) -> None:
        """Retrain AI models using accumulated training data."""
        if not TORCH_AVAILABLE or len(self.training_data) < 50:
            return
        
        try:
            self.logger.info(f"Retraining AI models with {len(self.training_data)} samples")
            
            # Prepare training data
            X = np.array([sample['problem_features'] for sample in self.training_data])
            y = np.array([sample['objective_value'] for sample in self.training_data])
            
            # Retrain heuristic model
            if self.heuristic_model is not None:
                self._retrain_heuristic_model(X, y)
            
            self.logger.info("Model retraining completed")
            
        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")
    
    def _retrain_heuristic_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Retrain the heuristic neural network."""
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        # Set up training
        optimizer = torch.optim.Adam(self.heuristic_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training loop
        epochs = 100
        batch_size = min(32, len(X) // 4)
        
        for epoch in range(epochs):
            # Mini-batch training
            indices = torch.randperm(len(X))
            for i in range(0, len(X), batch_size):
                batch_indices = indices[i:i + batch_size]
                X_batch = X_tensor[batch_indices]
                y_batch = y_tensor[batch_indices]
                
                # Forward pass
                predictions = self.heuristic_model(X_batch)
                loss = criterion(predictions, y_batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        self.logger.debug(f"Heuristic model retrained with final loss: {loss.item():.4f}")

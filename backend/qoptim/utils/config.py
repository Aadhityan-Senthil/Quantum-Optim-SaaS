"""
Configuration management for Q-Optim framework.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """
    Configuration management system for Q-Optim framework.
    
    Supports loading from YAML files, environment variables, and direct setting.
    """
    
    def __init__(self, config_file: Optional[str] = None, **kwargs):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to YAML configuration file
            **kwargs: Direct configuration parameters
        """
        self.config: Dict[str, Any] = {
            # Quantum settings
            'quantum_backend': 'qasm_simulator',
            'quantum_shots': 1024,
            'max_qubits': 20,
            'use_pennylane': False,
            
            # QAOA settings
            'qaoa_layers': 3,
            'qaoa_max_iterations': 500,
            'qaoa_tolerance': 1e-6,
            'qaoa_optimizer': 'COBYLA',
            'qaoa_warm_start': True,
            
            # VQE settings
            'vqe_layers': 2,
            'vqe_max_iterations': 300,
            'vqe_tolerance': 1e-6,
            'vqe_optimizer': 'COBYLA',
            'vqe_ansatz': 'RealAmplitudes',
            'vqe_entanglement': 'linear',
            'vqe_bits_per_variable': 3,
            
            # AI preprocessing settings
            'ai_feature_selection': True,
            'ai_problem_reduction': True,
            'ai_heuristic_learning': True,
            'ai_gnn_enabled': True,
            'ai_rl_enabled': True,
            'ai_pca_components': 0.95,
            'ai_feature_k': 50,
            'ai_gnn_hidden_dim': 64,
            'ai_rl_hidden_dim': 128,
            
            # Classical solver settings
            'sa_initial_temp': 1000.0,
            'sa_cooling_rate': 0.95,
            'sa_max_iterations': 10000,
            'ga_population_size': 100,
            'ga_generations': 500,
            'ga_crossover_rate': 0.8,
            'ga_mutation_rate': 0.1,
            'classical_timeout': 300,
            
            # Framework settings
            'solver_timeout': 300,
            'log_level': 'INFO',
            'parallel_workers': 4,
            'cache_enabled': True,
            'benchmarking_enabled': True,
            
            # Hardware settings
            'use_gpu': False,
            'device': 'cpu',
            'memory_limit': '8GB',
            
            # Monitoring settings
            'metrics_enabled': True,
            'telemetry_enabled': False,
            'prometheus_port': 9090,
            
            # Production settings
            'debug_mode': False,
            'profile_performance': False,
            'save_results': True,
            'results_directory': './results',
        }
        
        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)
        
        # Override with direct parameters
        self.config.update(kwargs)
        
        # Override with environment variables
        self._load_from_env()
    
    def load_from_file(self, config_file: str) -> None:
        """Load configuration from YAML file."""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            if file_config:
                self.config.update(file_config)
                
        except Exception as e:
            raise ValueError(f"Error loading configuration file: {e}")
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_prefix = 'QOPTIM_'
        
        for key in self.config.keys():
            env_key = f"{env_prefix}{key.upper()}"
            env_value = os.environ.get(env_key)
            
            if env_value is not None:
                # Convert string to appropriate type
                converted_value = self._convert_env_value(env_value, self.config[key])
                self.config[key] = converted_value
    
    def _convert_env_value(self, env_value: str, default_value: Any) -> Any:
        """Convert environment variable string to appropriate type."""
        if isinstance(default_value, bool):
            return env_value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(default_value, int):
            return int(env_value)
        elif isinstance(default_value, float):
            return float(env_value)
        else:
            return env_value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values."""
        self.config.update(updates)
    
    def save_to_file(self, config_file: str) -> None:
        """Save current configuration to YAML file."""
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ValueError(f"Error saving configuration file: {e}")
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return validation results."""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check quantum settings
        if self.config['quantum_shots'] <= 0:
            validation_results['errors'].append("quantum_shots must be positive")
        
        if self.config['max_qubits'] <= 0:
            validation_results['errors'].append("max_qubits must be positive")
        
        # Check QAOA settings
        if self.config['qaoa_layers'] <= 0:
            validation_results['errors'].append("qaoa_layers must be positive")
        
        if self.config['qaoa_max_iterations'] <= 0:
            validation_results['errors'].append("qaoa_max_iterations must be positive")
        
        # Check VQE settings
        if self.config['vqe_layers'] <= 0:
            validation_results['errors'].append("vqe_layers must be positive")
        
        # Check classical solver settings
        if self.config['sa_initial_temp'] <= 0:
            validation_results['errors'].append("sa_initial_temp must be positive")
        
        if self.config['ga_population_size'] <= 0:
            validation_results['errors'].append("ga_population_size must be positive")
        
        # Check timeouts
        if self.config['solver_timeout'] <= 0:
            validation_results['errors'].append("solver_timeout must be positive")
        
        # Warnings for potentially suboptimal settings
        if self.config['quantum_shots'] < 100:
            validation_results['warnings'].append("Low quantum_shots may lead to poor results")
        
        if self.config['qaoa_layers'] > 10:
            validation_results['warnings'].append("High qaoa_layers may be slow on NISQ devices")
        
        # Set validation status
        validation_results['valid'] = len(validation_results['errors']) == 0
        
        return validation_results
    
    def get_quantum_config(self) -> Dict[str, Any]:
        """Get quantum-specific configuration."""
        quantum_keys = [k for k in self.config.keys() if k.startswith(('quantum_', 'qaoa_', 'vqe_'))]
        return {k: self.config[k] for k in quantum_keys}
    
    def get_ai_config(self) -> Dict[str, Any]:
        """Get AI-specific configuration."""
        ai_keys = [k for k in self.config.keys() if k.startswith('ai_')]
        return {k: self.config[k] for k in ai_keys}
    
    def get_classical_config(self) -> Dict[str, Any]:
        """Get classical solver configuration."""
        classical_keys = [k for k in self.config.keys() if k.startswith(('sa_', 'ga_', 'classical_'))]
        return {k: self.config[k] for k in classical_keys}
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-style setting."""
        self.config[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return key in self.config
    
    def keys(self):
        """Get configuration keys."""
        return self.config.keys()
    
    def items(self):
        """Get configuration items."""
        return self.config.items()
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self.config.copy()
    
    def __str__(self) -> str:
        """String representation."""
        return f"Config({len(self.config)} parameters)"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Config({self.config})"


# Default configuration instance
default_config = Config()


def load_config(config_file: Optional[str] = None, **kwargs) -> Config:
    """
    Load configuration from file or create with parameters.
    
    Args:
        config_file: Path to YAML configuration file
        **kwargs: Direct configuration parameters
        
    Returns:
        Config instance
    """
    return Config(config_file, **kwargs)


def create_config_template(output_file: str) -> None:
    """
    Create a configuration template file with all available options.
    
    Args:
        output_file: Path where to save the template
    """
    template_config = Config()
    
    # Add comments to the template
    template_dict = template_config.to_dict()
    
    # Create commented YAML structure
    yaml_content = """# Q-Optim Configuration Template
# Quantum Computing Settings
quantum_backend: qasm_simulator  # 'qasm_simulator', 'ibmq_qasm_simulator', or real hardware
quantum_shots: 1024  # Number of shots for quantum measurements
max_qubits: 20  # Maximum number of qubits for problems
use_pennylane: false  # Use PennyLane instead of Qiskit for quantum circuits

# QAOA Settings
qaoa_layers: 3  # Number of QAOA layers (p parameter)
qaoa_max_iterations: 500  # Maximum optimization iterations
qaoa_tolerance: 1e-6  # Convergence tolerance
qaoa_optimizer: COBYLA  # Classical optimizer: COBYLA, SPSA, L_BFGS_B
qaoa_warm_start: true  # Use classical warm-start

# VQE Settings  
vqe_layers: 2  # Number of ansatz layers
vqe_max_iterations: 300  # Maximum optimization iterations
vqe_tolerance: 1e-6  # Convergence tolerance
vqe_optimizer: COBYLA  # Classical optimizer
vqe_ansatz: RealAmplitudes  # Ansatz type: RealAmplitudes, EfficientSU2, UCCSD
vqe_entanglement: linear  # Entanglement pattern
vqe_bits_per_variable: 3  # Bits for discretizing continuous variables

# AI Preprocessing Settings
ai_feature_selection: true  # Enable feature selection
ai_problem_reduction: true  # Enable problem reduction techniques
ai_heuristic_learning: true  # Enable heuristic neural networks
ai_gnn_enabled: true  # Enable graph neural networks
ai_rl_enabled: true  # Enable reinforcement learning
ai_pca_components: 0.95  # PCA variance threshold
ai_feature_k: 50  # Number of features to select
ai_gnn_hidden_dim: 64  # GNN hidden layer dimension
ai_rl_hidden_dim: 128  # RL network hidden dimension

# Classical Solver Settings
sa_initial_temp: 1000.0  # Simulated annealing initial temperature
sa_cooling_rate: 0.95  # Cooling rate
sa_max_iterations: 10000  # Maximum SA iterations
ga_population_size: 100  # Genetic algorithm population size
ga_generations: 500  # Number of GA generations
ga_crossover_rate: 0.8  # Crossover probability
ga_mutation_rate: 0.1  # Mutation probability
classical_timeout: 300  # Timeout for classical methods (seconds)

# Framework Settings
solver_timeout: 300  # Global solver timeout (seconds)
log_level: INFO  # Logging level: DEBUG, INFO, WARNING, ERROR
parallel_workers: 4  # Number of parallel workers
cache_enabled: true  # Enable solution caching
benchmarking_enabled: true  # Enable benchmarking against classical methods

# Hardware Settings
use_gpu: false  # Use GPU acceleration where available
device: cpu  # Device: cpu, cuda, auto
memory_limit: 8GB  # Memory limit

# Monitoring Settings
metrics_enabled: true  # Enable metrics collection
telemetry_enabled: false  # Enable telemetry
prometheus_port: 9090  # Prometheus metrics port

# Production Settings
debug_mode: false  # Enable debug mode
profile_performance: false  # Enable performance profiling
save_results: true  # Save optimization results
results_directory: ./results  # Results output directory
"""
    
    try:
        with open(output_file, 'w') as f:
            f.write(yaml_content)
    except Exception as e:
        raise ValueError(f"Error creating configuration template: {e}")

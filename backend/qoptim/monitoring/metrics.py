"""
Metrics collection and monitoring for Q-Optim framework.
"""

import time
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import threading
from dataclasses import dataclass
from datetime import datetime


@dataclass 
class Metric:
    """A single metric data point."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str]


class MetricsCollector:
    """
    Lightweight metrics collection system for Q-Optim framework.
    
    Collects performance metrics, solution quality metrics, and system metrics.
    """
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, float] = {}
        self.lock = threading.Lock()
        
        # Performance tracking
        self.solve_times: Dict[str, List[float]] = defaultdict(list)
        self.success_counts: Dict[str, int] = defaultdict(int)
        self.failure_counts: Dict[str, int] = defaultdict(int)
        
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        with self.lock:
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics[name].append(metric)
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter."""
        with self.lock:
            self.counters[name] += value
            
        # Also record as time series
        self.record_metric(f"{name}_count", self.counters[name], tags)
    
    def record_solve_time(self, problem_name: str, solve_time: float) -> None:
        """Record problem solving time."""
        with self.lock:
            self.solve_times[problem_name].append(solve_time)
            
        self.record_metric("solve_time", solve_time, {"problem": problem_name})
    
    def record_success(self, solver_type: str, problem_type: str) -> None:
        """Record successful solve."""
        key = f"{solver_type}_{problem_type}"
        with self.lock:
            self.success_counts[key] += 1
            
        self.record_metric("success_rate", 1.0, {
            "solver": solver_type,
            "problem_type": problem_type
        })
    
    def record_failure(self, solver_type: str, problem_type: str) -> None:
        """Record failed solve."""
        key = f"{solver_type}_{problem_type}"
        with self.lock:
            self.failure_counts[key] += 1
            
        self.record_metric("success_rate", 0.0, {
            "solver": solver_type, 
            "problem_type": problem_type
        })
    
    def start_timer(self, name: str) -> None:
        """Start a timer."""
        with self.lock:
            self.timers[name] = time.time()
    
    def stop_timer(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Stop a timer and record elapsed time."""
        with self.lock:
            if name not in self.timers:
                return 0.0
                
            elapsed = time.time() - self.timers[name]
            del self.timers[name]
            
        self.record_metric(name, elapsed, tags)
        return elapsed
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        with self.lock:
            if name not in self.metrics:
                return {}
                
            values = [m.value for m in self.metrics[name]]
            
            if not values:
                return {}
                
            import numpy as np
            
            return {
                "count": len(values),
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "p50": np.percentile(values, 50),
                "p95": np.percentile(values, 95),
                "p99": np.percentile(values, 99)
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        with self.lock:
            return {
                "metrics": {name: list(metrics) for name, metrics in self.metrics.items()},
                "counters": dict(self.counters),
                "solve_times": dict(self.solve_times),
                "success_counts": dict(self.success_counts),
                "failure_counts": dict(self.failure_counts)
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "solver_performance": {},
            "problem_performance": {}
        }
        
        # Overall summary
        with self.lock:
            total_problems = len(self.solve_times)
            total_solves = sum(len(times) for times in self.solve_times.values())
            
            if total_solves > 0:
                all_times = []
                for times in self.solve_times.values():
                    all_times.extend(times)
                
                import numpy as np
                report["summary"] = {
                    "total_problems": total_problems,
                    "total_solves": total_solves,
                    "avg_solve_time": np.mean(all_times),
                    "total_successes": sum(self.success_counts.values()),
                    "total_failures": sum(self.failure_counts.values()),
                    "overall_success_rate": sum(self.success_counts.values()) / (
                        sum(self.success_counts.values()) + sum(self.failure_counts.values())
                    ) if (sum(self.success_counts.values()) + sum(self.failure_counts.values())) > 0 else 0.0
                }
        
        return report
    
    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        with self.lock:
            self.metrics.clear()
            self.counters.clear()
            self.timers.clear()
            self.solve_times.clear()
            self.success_counts.clear()
            self.failure_counts.clear()


# Global metrics collector instance
global_metrics = MetricsCollector()

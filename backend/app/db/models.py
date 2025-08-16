"""
QuantumOptim by AYNX AI - Database Models
Enterprise-grade database models for global SaaS platform
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON, Text, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from datetime import datetime
import uuid

Base = declarative_base()

class SubscriptionTier(enum.Enum):
    """Subscription tier enumeration"""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class OptimizationStatus(enum.Enum):
    """Optimization job status"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ProblemType(enum.Enum):
    """Optimization problem types"""
    MAX_CUT = "max_cut"
    KNAPSACK = "knapsack"
    TSP = "tsp"
    PORTFOLIO = "portfolio"
    SCHEDULING = "scheduling"
    GRAPH_COLORING = "graph_coloring"
    SUPPLY_CHAIN = "supply_chain"
    CUSTOM = "custom"

class User(Base):
    """Enterprise user model with global support"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=True)
    company = Column(String(100), nullable=True)
    
    # Authentication
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    verification_token = Column(String(255), nullable=True)
    
    # Subscription
    subscription_tier = Column(Enum(SubscriptionTier), default=SubscriptionTier.FREE)
    stripe_customer_id = Column(String(100), nullable=True)
    subscription_id = Column(String(100), nullable=True)
    subscription_status = Column(String(20), default="inactive")
    
    # Usage tracking
    monthly_optimization_count = Column(Integer, default=0)
    last_optimization_reset = Column(DateTime, default=func.now())
    total_optimizations = Column(Integer, default=0)
    
    # Global user data
    country = Column(String(2), nullable=True)  # ISO country code
    timezone = Column(String(50), default="UTC")
    preferred_language = Column(String(5), default="en")
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_login_at = Column(DateTime, nullable=True)
    
    # Relationships
    optimization_jobs = relationship("OptimizationJob", back_populates="user")
    api_keys = relationship("APIKey", back_populates="user")
    billing_history = relationship("BillingHistory", back_populates="user")
    
    def get_optimization_limit(self):
        """Get monthly optimization limit based on subscription tier"""
        limits = {
            SubscriptionTier.FREE: 10,
            SubscriptionTier.PRO: 1000,
            SubscriptionTier.ENTERPRISE: float('inf')
        }
        return limits.get(self.subscription_tier, 10)
    
    def can_create_optimization(self):
        """Check if user can create a new optimization job"""
        return self.monthly_optimization_count < self.get_optimization_limit()

class APIKey(Base):
    """API key management for enterprise customers"""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key_id = Column(String(32), unique=True, index=True, nullable=False)
    key_secret = Column(String(64), nullable=False)  # Hashed
    
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Rate limiting
    requests_per_minute = Column(Integer, default=60)
    daily_request_limit = Column(Integer, default=1000)
    current_daily_requests = Column(Integer, default=0)
    
    # Permissions
    can_submit_jobs = Column(Boolean, default=True)
    can_view_results = Column(Boolean, default=True)
    can_cancel_jobs = Column(Boolean, default=True)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    last_used_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    
    # Relationship
    user = relationship("User", back_populates="api_keys")

class OptimizationJob(Base):
    """Optimization job tracking"""
    __tablename__ = "optimization_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Problem definition
    problem_name = Column(String(200), nullable=False)
    problem_type = Column(Enum(ProblemType), nullable=False)
    problem_data = Column(JSON, nullable=False)
    
    # Algorithm selection
    algorithms = Column(JSON, nullable=False)  # List of algorithms to use
    use_ai_preprocessing = Column(Boolean, default=True)
    max_execution_time = Column(Integer, default=300)  # seconds
    
    # Job status
    status = Column(Enum(OptimizationStatus), default=OptimizationStatus.QUEUED)
    progress = Column(Float, default=0.0)  # 0.0 to 1.0
    
    # Results
    result_data = Column(JSON, nullable=True)
    best_objective_value = Column(Float, nullable=True)
    solution_variables = Column(JSON, nullable=True)
    algorithm_used = Column(String(50), nullable=True)
    
    # Performance metrics
    execution_time = Column(Float, nullable=True)  # seconds
    iterations_completed = Column(Integer, nullable=True)
    quantum_advantage_detected = Column(Boolean, default=False)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Cost tracking (for billing)
    compute_cost = Column(Float, default=0.0)  # in USD cents
    
    # Relationship
    user = relationship("User", back_populates="optimization_jobs")

class BillingHistory(Base):
    """Billing and payment history"""
    __tablename__ = "billing_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Stripe data
    stripe_payment_intent_id = Column(String(100), nullable=True)
    stripe_invoice_id = Column(String(100), nullable=True)
    
    # Payment details
    amount = Column(Integer, nullable=False)  # in cents
    currency = Column(String(3), default="USD")
    description = Column(String(200), nullable=True)
    
    # Status
    status = Column(String(20), nullable=False)  # succeeded, failed, pending
    payment_method = Column(String(20), nullable=True)  # card, bank_transfer, etc.
    
    # Billing period
    billing_period_start = Column(DateTime, nullable=True)
    billing_period_end = Column(DateTime, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    paid_at = Column(DateTime, nullable=True)
    
    # Relationship
    user = relationship("User", back_populates="billing_history")

class SystemMetrics(Base):
    """System-wide metrics for monitoring and analytics"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Metrics data
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20), nullable=True)
    
    # Dimensions
    region = Column(String(20), nullable=True)
    service = Column(String(50), nullable=True)
    user_tier = Column(String(20), nullable=True)
    
    # Metadata
    recorded_at = Column(DateTime, default=func.now())
    tags = Column(JSON, nullable=True)

class CompanyProfile(Base):
    """Company profiles for enterprise customers"""
    __tablename__ = "company_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    company_name = Column(String(200), nullable=False)
    industry = Column(String(100), nullable=True)
    company_size = Column(String(20), nullable=True)  # startup, small, medium, large, enterprise
    
    # Contact information
    primary_contact_email = Column(String(255), nullable=False)
    billing_email = Column(String(255), nullable=True)
    phone_number = Column(String(20), nullable=True)
    
    # Address
    address_line1 = Column(String(200), nullable=True)
    address_line2 = Column(String(200), nullable=True)
    city = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    postal_code = Column(String(20), nullable=True)
    country = Column(String(2), nullable=True)  # ISO country code
    
    # Tax information
    tax_id = Column(String(50), nullable=True)
    vat_number = Column(String(50), nullable=True)
    
    # Enterprise features
    sso_enabled = Column(Boolean, default=False)
    api_access_enabled = Column(Boolean, default=False)
    priority_support = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class AuditLog(Base):
    """Audit log for compliance and security"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Action details
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(100), nullable=True)
    
    # Request details
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(500), nullable=True)
    
    # Metadata
    details = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=func.now())
    
    # Compliance
    data_classification = Column(String(20), default="internal")  # public, internal, confidential, restricted

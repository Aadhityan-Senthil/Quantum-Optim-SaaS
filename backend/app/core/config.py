"""
QuantumOptim by AYNX AI - Configuration Settings
Enterprise-grade configuration for global deployment
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, validator
import secrets

class Settings(BaseSettings):
    """Application settings with enterprise-grade configuration"""
    
    # Application
    APP_NAME: str = "QuantumOptim by AYNX AI"
    APP_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # Company Information
    COMPANY_NAME: str = "AYNX AI"
    COMPANY_WEBSITE: str = "https://aynx.ai"
    COMPANY_EMAIL: str = "support@aynx.ai"
    COMPANY_HEADQUARTERS: str = "India"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "sqlite+aiosqlite:///./app.db"
    )
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # CORS Origins for global access
    CORS_ORIGINS: List[str] = [
        "https://quantumoptim.aynx.ai",
        "https://www.quantumoptim.aynx.ai",
        "https://app.quantumoptim.aynx.ai",
        "http://localhost:3000",  # Development
        "http://localhost:8080",  # Development
    ]
    
    # Trusted hosts for security
    ALLOWED_HOSTS: List[str] = [
        "quantumoptim.aynx.ai",
        "api.quantumoptim.aynx.ai",
        "*.railway.app",
        "localhost",
        "127.0.0.1"
    ]
    
    # Email Configuration (for global notifications)
    SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int = 587
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    
    # Payment Configuration (Razorpay for India, Stripe for Global)
    PAYMENT_PROVIDER: str = os.getenv("PAYMENT_PROVIDER", "razorpay")  # razorpay or stripe
    
    # Razorpay Configuration (Primary for India)
    RAZORPAY_KEY_ID: str = os.getenv("RAZORPAY_KEY_ID", "")
    RAZORPAY_KEY_SECRET: str = os.getenv("RAZORPAY_KEY_SECRET", "")
    RAZORPAY_WEBHOOK_SECRET: str = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")
    
    # Stripe Configuration (Backup for Global)
    STRIPE_PUBLISHABLE_KEY: str = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
    STRIPE_SECRET_KEY: str = os.getenv("STRIPE_SECRET_KEY", "")
    STRIPE_WEBHOOK_SECRET: str = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    
    # Subscription Tiers (in cents for international compatibility)
    PRICING_FREE_LIMIT: int = 10  # optimizations per month
    PRICING_PRO_MONTHLY: int = 2900  # $29.00 USD
    PRICING_PRO_LIMIT: int = 1000  # optimizations per month
    PRICING_ENTERPRISE_MONTHLY: int = 19900  # $199.00 USD
    
    # Rate Limiting (requests per minute)
    RATE_LIMIT_FREE: int = 10
    RATE_LIMIT_PRO: int = 100
    RATE_LIMIT_ENTERPRISE: int = 1000
    
    # Celery Configuration
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
    
    # Monitoring and Logging
    SENTRY_DSN: Optional[str] = os.getenv("SENTRY_DSN")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Feature Flags for global rollout
    FEATURE_QUANTUM_ALGORITHMS: bool = True
    FEATURE_AI_PREPROCESSING: bool = True
    FEATURE_REAL_TIME_OPTIMIZATION: bool = True
    FEATURE_ENTERPRISE_SUPPORT: bool = True
    
    # Performance Settings
    MAX_CONCURRENT_JOBS_FREE: int = 1
    MAX_CONCURRENT_JOBS_PRO: int = 5
    MAX_CONCURRENT_JOBS_ENTERPRISE: int = 50
    
    # Global Support
    SUPPORT_TIMEZONES: List[str] = [
        "Asia/Kolkata",    # India (HQ)
        "America/New_York",  # US East
        "America/Los_Angeles",  # US West
        "Europe/London",      # UK
        "Asia/Tokyo",        # Japan
        "Australia/Sydney"   # Australia
    ]
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        if v not in ["development", "staging", "production"]:
            raise ValueError("Environment must be development, staging, or production")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# SEO and Marketing Configuration
class SEOConfig:
    """SEO configuration for global reach"""
    
    SITE_NAME = "QuantumOptim by AYNX AI"
    SITE_DESCRIPTION = (
        "Solve complex business problems 10x faster with quantum-enhanced AI. "
        "Trusted by Fortune 500 companies for supply chain optimization, financial "
        "portfolio management, and resource scheduling worldwide."
    )
    
    # Primary keywords for SEO
    KEYWORDS = [
        "quantum optimization",
        "AI optimization",
        "supply chain optimization",
        "portfolio optimization",
        "business optimization software",
        "quantum computing business",
        "optimization as a service",
        "enterprise optimization",
        "quantum algorithms",
        "optimization API"
    ]
    
    # Regional keywords for global SEO
    REGIONAL_KEYWORDS = {
        "india": ["quantum optimization india", "AI startup india", "optimization software india"],
        "us": ["quantum computing usa", "optimization saas usa", "business optimization america"],
        "europe": ["quantum optimization europe", "AI optimization eu", "business software europe"],
        "asia": ["quantum computing asia", "optimization platform asia", "enterprise software asia"]
    }
    
    # Social media and branding
    SOCIAL_MEDIA = {
        "twitter": "@AYNXAI",
        "linkedin": "company/aynx-ai",
        "youtube": "c/AYNXAI",
        "github": "Aadhityan-Senthil"
    }
    
    # Open Graph and Twitter Card data
    OG_IMAGE = "https://aynx.ai/images/quantumoptim-social.png"
    OG_LOCALE = "en_US"
    TWITTER_CREATOR = "@AadhityanSenthil"

seo_config = SEOConfig()

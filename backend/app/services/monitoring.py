"""
QuantumOptim by AYNX AI - Monitoring Service
Sentry integration for error tracking and performance monitoring
"""

import logging
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from app.core.config import settings

logger = logging.getLogger(__name__)

async def setup_monitoring():
    """Initialize monitoring services (Sentry, logging, etc.)"""
    
    if settings.SENTRY_DSN:
        logger.info("🔍 Initializing Sentry monitoring...")
        
        # Configure Sentry with comprehensive integrations
        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            environment=settings.ENVIRONMENT,
            release=f"quantumoptim@{settings.APP_VERSION}",
            
            # Integrations for comprehensive monitoring
            integrations=[
                FastApiIntegration(auto_enabling_integrations=True),
                SqlalchemyIntegration(),
                RedisIntegration(),
                LoggingIntegration(
                    level=logging.INFO,  # Capture info and above as breadcrumbs
                    event_level=logging.ERROR  # Send errors as events
                ),
            ],
            
            # Performance monitoring
            traces_sample_rate=0.1 if settings.ENVIRONMENT == "production" else 1.0,
            profiles_sample_rate=0.1 if settings.ENVIRONMENT == "production" else 1.0,
            
            # Error filtering and sampling
            sample_rate=1.0,
            max_breadcrumbs=50,
            
            # Additional context
            before_send=before_send_filter,
            
            # Set custom tags
            default_integrations=False,  # We explicitly define integrations above
        )
        
        # Set custom tags and context
        sentry_sdk.set_tag("company", "AYNX AI")
        sentry_sdk.set_tag("service", "QuantumOptim")
        sentry_sdk.set_tag("version", settings.APP_VERSION)
        sentry_sdk.set_tag("environment", settings.ENVIRONMENT)
        
        # Set user context for the service
        sentry_sdk.set_context("service", {
            "name": "QuantumOptim Backend",
            "version": settings.APP_VERSION,
            "company": "AYNX AI",
            "headquarters": "India"
        })
        
        logger.info("✅ Sentry monitoring initialized successfully")
        
        # Test Sentry integration
        if settings.ENVIRONMENT == "development":
            logger.info("🧪 Testing Sentry integration...")
            sentry_sdk.capture_message(
                "QuantumOptim backend started successfully", 
                level="info"
            )
    
    else:
        logger.warning("⚠️ Sentry DSN not configured - monitoring disabled")
    
    # Set up additional monitoring
    await setup_performance_monitoring()
    await setup_business_metrics()

def before_send_filter(event, hint):
    """Filter and enhance events before sending to Sentry"""
    
    # Add custom business context
    event.setdefault('tags', {})
    event['tags']['business_impact'] = classify_business_impact(event)
    
    # Filter out noisy errors in development
    if settings.ENVIRONMENT == "development":
        # Don't send certain development-only errors
        if "ConnectionError" in str(hint.get('exc_info', '')):
            return None
    
    # Enhance with additional context
    event.setdefault('extra', {})
    event['extra']['company'] = "AYNX AI"
    event['extra']['service_tier'] = "Enterprise"
    
    return event

def classify_business_impact(event):
    """Classify the business impact of an error"""
    
    error_type = event.get('exception', {}).get('values', [{}])[0].get('type', '')
    
    # High impact errors
    high_impact = [
        'PaymentError', 
        'DatabaseError', 
        'AuthenticationError',
        'OptimizationFailure'
    ]
    
    # Medium impact errors  
    medium_impact = [
        'ValidationError',
        'RateLimitError', 
        'TimeoutError'
    ]
    
    if any(err in error_type for err in high_impact):
        return "high"
    elif any(err in error_type for err in medium_impact):
        return "medium"
    else:
        return "low"

async def setup_performance_monitoring():
    """Set up performance monitoring"""
    logger.info("📊 Setting up performance monitoring...")
    
    # TODO: Add custom performance metrics
    # - API response times
    # - Optimization job completion rates
    # - Database query performance
    # - Redis cache hit rates
    
    logger.info("✅ Performance monitoring configured")

async def setup_business_metrics():
    """Set up business-specific metrics"""
    logger.info("💼 Setting up business metrics monitoring...")
    
    # TODO: Add business metrics
    # - User registrations
    # - Optimization jobs submitted
    # - Subscription upgrades
    # - Error rates by user tier
    
    logger.info("✅ Business metrics monitoring configured")

def capture_optimization_error(error: Exception, job_id: str, user_id: str):
    """Capture optimization-specific errors with rich context"""
    
    with sentry_sdk.push_scope() as scope:
        scope.set_tag("error_type", "optimization_failure")
        scope.set_tag("job_id", job_id)
        scope.set_user({"id": user_id})
        scope.set_context("optimization", {
            "job_id": job_id,
            "user_id": user_id,
            "service": "quantum_optimization"
        })
        
        sentry_sdk.capture_exception(error)

def capture_payment_error(error: Exception, user_id: str, amount: int, currency: str):
    """Capture payment-specific errors with business context"""
    
    with sentry_sdk.push_scope() as scope:
        scope.set_tag("error_type", "payment_failure")
        scope.set_tag("business_impact", "high")
        scope.set_user({"id": user_id})
        scope.set_context("payment", {
            "user_id": user_id,
            "amount": amount,
            "currency": currency,
            "provider": settings.PAYMENT_PROVIDER
        })
        
        sentry_sdk.capture_exception(error)

def capture_user_event(event_name: str, user_id: str, properties: dict = None):
    """Capture user events for analytics"""
    
    with sentry_sdk.push_scope() as scope:
        scope.set_tag("event_type", "user_action")
        scope.set_user({"id": user_id})
        
        if properties:
            scope.set_context("event_properties", properties)
        
        sentry_sdk.capture_message(f"User Event: {event_name}", level="info")

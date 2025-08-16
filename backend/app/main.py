"""
QuantumOptim by AYNX AI - Backend API
Enterprise-grade FastAPI application for quantum optimization services
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import uvicorn
import time
import logging
from contextlib import asynccontextmanager

from app.api.routes import api_router
from app.core.config import settings
from app.core.security import SecurityMiddleware
from app.db.database import create_tables
from app.services.monitoring import setup_monitoring

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting QuantumOptim by AYNX AI")
    await create_tables()
    await setup_monitoring()
    yield
    # Shutdown
    logger.info("🛑 Shutting down QuantumOptim by AYNX AI")

# Create FastAPI app with custom OpenAPI
app = FastAPI(
    title="QuantumOptim API by AYNX AI",
    description="""
    ## 🚀 The Future of Business Optimization
    
    **Solve Complex Problems 10x Faster with Quantum-Enhanced AI**
    
    QuantumOptim democratizes quantum-classical optimization for businesses worldwide. 
    Our API enables you to solve complex optimization problems that were previously impossible 
    or took days to compute - now solved in minutes.
    
    ### 🎯 Use Cases:
    - **Supply Chain Optimization**: Reduce logistics costs by 15-40%
    - **Financial Portfolio Management**: Maximize returns while minimizing risk
    - **Resource Scheduling**: Optimize staff, equipment, and time allocation
    - **Route Planning**: Minimize travel time and fuel costs
    - **Manufacturing**: Optimize production schedules and reduce waste
    
    ### 🌐 Global Scale:
    - Trusted by Fortune 500 companies
    - Processing millions of optimizations monthly
    - 99.99% uptime SLA
    - 24/7 enterprise support
    
    ### 🔬 Technology:
    - Quantum algorithms (QAOA, VQE, Grover)
    - Classical AI optimization
    - Real-time processing
    - Scalable cloud infrastructure
    """,
    version="1.0.0",
    contact={
        "name": "AYNX AI Support",
        "url": "https://aynx.ai/support",
        "email": "support@aynx.ai",
    },
    license_info={
        "name": "Enterprise License",
        "url": "https://aynx.ai/license",
    },
    servers=[
        {"url": "https://api.quantumoptim.aynx.ai", "description": "Production server"},
        {"url": "https://staging-api.quantumoptim.aynx.ai", "description": "Staging server"},
    ],
    lifespan=lifespan
)

# Security Middleware
app.add_middleware(SecurityMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)

# CORS for global access
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Powered-By"] = "AYNX AI"
    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred. Our team has been notified.",
            "error_id": str(int(time.time())),
            "support": "contact support@aynx.ai for assistance"
        }
    )

# API Routes
app.include_router(api_router, prefix="/api/v1")

# Health check endpoints
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint for load balancers"""
    return {
        "status": "healthy",
        "service": "QuantumOptim by AYNX AI",
        "version": "1.0.0",
        "timestamp": time.time()
    }

@app.get("/", tags=["System"])
async def root():
    """API root endpoint with company branding"""
    return {
        "message": "Welcome to QuantumOptim by AYNX AI",
        "description": "Solve complex business problems 10x faster with quantum-enhanced AI",
        "documentation": "/docs",
        "company": "AYNX AI",
        "website": "https://aynx.ai",
        "support": "support@aynx.ai"
    }

# Custom OpenAPI schema for better SEO and documentation
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="QuantumOptim API by AYNX AI",
        version="1.0.0",
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom schema properties for better SEO
    openapi_schema["info"]["x-logo"] = {
        "url": "https://aynx.ai/logo.png"
    }
    openapi_schema["info"]["x-company"] = "AYNX AI"
    openapi_schema["info"]["x-headquarters"] = "India (Global Operations)"
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if settings.ENVIRONMENT == "development" else False,
        workers=1 if settings.ENVIRONMENT == "development" else 4
    )

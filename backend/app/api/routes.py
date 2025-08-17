"""
API router for QuantumOptim by AYNX AI
Exposes health and placeholder endpoints to verify deployment.
"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/status", tags=["System"])
async def api_status():
    return {"status": "ok"}

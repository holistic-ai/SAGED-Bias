from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional
from app_simplified.backend.services.database_service import DatabaseService

router = APIRouter(
    prefix="/db",
    tags=["database"],
    responses={404: {"description": "Not found"}},
)

# Create a single instance of DatabaseService
db_service = DatabaseService()

@router.get("/benchmark/{domain}", response_model=Dict[str, Any])
async def get_benchmark(domain: str):
    """Get benchmark data for a specific domain"""
    try:
        data = db_service.get_latest_data('benchmark', domain)
        if not data:
            raise HTTPException(status_code=404, detail="Benchmark not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/benchmark/{domain}/keywords", response_model=Dict[str, Any])
async def get_keywords(domain: str):
    """Get keywords for a specific domain"""
    try:
        data = db_service.get_latest_data('keywords', domain)
        if not data:
            raise HTTPException(status_code=404, detail="Keywords not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/benchmark/{domain}/source-finder", response_model=Dict[str, Any])
async def get_source_finder(domain: str):
    """Get source finder data for a specific domain"""
    try:
        data = db_service.get_latest_data('source_finder', domain)
        if not data:
            raise HTTPException(status_code=404, detail="Source finder data not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/benchmark/{domain}/scraped-sentences", response_model=Dict[str, Any])
async def get_scraped_sentences(domain: str):
    """Get scraped sentences for a specific domain"""
    try:
        data = db_service.get_latest_data('scraped_sentences', domain)
        if not data:
            raise HTTPException(status_code=404, detail="Scraped sentences not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/benchmark/{domain}/status", response_model=Dict[str, Any])
async def get_benchmark_status(domain: str):
    """Get the status of all benchmark-related tables for a domain"""
    try:
        status = {
            "benchmark": False,
            "keywords": False,
            "source_finder": False,
            "scraped_sentences": False
        }
        
        for tier in status.keys():
            data = db_service.get_latest_data(tier, domain)
            status[tier] = data is not None
        
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
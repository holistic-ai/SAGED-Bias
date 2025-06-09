from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from app_simplified.backend.services.database_service import DatabaseService
from app_simplified.backend.schemas.build_config import (
    DomainBenchmarkConfig, BenchmarkResponse,
    KeywordsData, SourceFinderData, ScrapedSentencesData,
    SplitSentencesData, QuestionsData, ReplacementDescriptionData,
    AllDataTiersResponse
)
from app_simplified.backend.database import get_db
from app_simplified.backend.services.saged_service import SagedService

router = APIRouter(
    prefix="/benchmark",
    tags=["benchmark"]
)

@router.post("/build", response_model=BenchmarkResponse)
async def build_benchmark(
    config: DomainBenchmarkConfig,
    db: Session = Depends(get_db)
):
    """Build a benchmark with the given configuration"""
    try:
        saged_service = SagedService()
        # Get database config for SAGED
        db_config = saged_service.db_service.get_database_config()
        
        # Update config with database settings
        config.database_config = db_config
        
        # Call SAGED pipeline with the config
        return await saged_service.build_benchmark(config.domain, config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/keywords/{domain}", response_model=List[KeywordsData])
async def get_keywords(
    domain: str,
    db: Session = Depends(get_db)
):
    """Get all keywords data for a domain"""
    try:
        db_service = DatabaseService()
        return db_service.get_keywords(domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/keywords/{domain}/latest", response_model=Optional[KeywordsData])
async def get_latest_keywords(
    domain: str,
    db: Session = Depends(get_db)
):
    """Get the latest keywords data for a domain"""
    try:
        db_service = DatabaseService()
        return db_service.get_latest_keywords(domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/source-finder/{domain}", response_model=List[SourceFinderData])
async def get_source_finder(
    domain: str,
    db: Session = Depends(get_db)
):
    """Get all source finder data for a domain"""
    try:
        db_service = DatabaseService()
        return db_service.get_source_finder(domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/source-finder/{domain}/latest", response_model=Optional[SourceFinderData])
async def get_latest_source_finder(
    domain: str,
    db: Session = Depends(get_db)
):
    """Get the latest source finder data for a domain"""
    try:
        db_service = DatabaseService()
        return db_service.get_latest_source_finder(domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scraped-sentences/{domain}", response_model=List[ScrapedSentencesData])
async def get_scraped_sentences(
    domain: str,
    db: Session = Depends(get_db)
):
    """Get all scraped sentences data for a domain"""
    try:
        db_service = DatabaseService()
        return db_service.get_scraped_sentences(domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scraped-sentences/{domain}/latest", response_model=Optional[ScrapedSentencesData])
async def get_latest_scraped_sentences(
    domain: str,
    db: Session = Depends(get_db)
):
    """Get the latest scraped sentences data for a domain"""
    try:
        db_service = DatabaseService()
        return db_service.get_latest_scraped_sentences(domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/split-sentences/{domain}", response_model=List[SplitSentencesData])
async def get_split_sentences(
    domain: str,
    db: Session = Depends(get_db)
):
    """Get all split sentences data for a domain"""
    try:
        db_service = DatabaseService()
        return db_service.get_split_sentences(domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/split-sentences/{domain}/latest", response_model=Optional[SplitSentencesData])
async def get_latest_split_sentences(
    domain: str,
    db: Session = Depends(get_db)
):
    """Get the latest split sentences data for a domain"""
    try:
        db_service = DatabaseService()
        return db_service.get_latest_split_sentences(domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/questions/{domain}", response_model=List[QuestionsData])
async def get_questions(
    domain: str,
    db: Session = Depends(get_db)
):
    """Get all questions data for a domain"""
    try:
        db_service = DatabaseService()
        return db_service.get_questions(domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/questions/{domain}/latest", response_model=Optional[QuestionsData])
async def get_latest_questions(
    domain: str,
    db: Session = Depends(get_db)
):
    """Get the latest questions data for a domain"""
    try:
        db_service = DatabaseService()
        return db_service.get_latest_questions(domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/replacement-description/{domain}", response_model=List[ReplacementDescriptionData])
async def get_replacement_description(
    domain: str,
    db: Session = Depends(get_db)
):
    """Get all replacement description data for a domain"""
    try:
        db_service = DatabaseService()
        return db_service.get_replacement_description(domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/replacement-description/{domain}/latest", response_model=Optional[ReplacementDescriptionData])
async def get_latest_replacement_description(
    domain: str,
    db: Session = Depends(get_db)
):
    """Get the latest replacement description data for a domain"""
    try:
        db_service = DatabaseService()
        return db_service.get_latest_replacement_description(domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/all/{domain}", response_model=AllDataTiersResponse)
async def get_all_data_tiers(
    domain: str,
    db: Session = Depends(get_db)
):
    """Get all data tiers for a domain"""
    try:
        db_service = DatabaseService()
        return db_service.get_all_data_tiers(domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
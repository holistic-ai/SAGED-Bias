from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
import uuid
from sqlalchemy.orm import Session

# Import with fallback for direct execution
try:
    from ..database import get_db
    from ..services import SAGEDService
except ImportError:
    from database import get_db
    from services import SAGEDService

router = APIRouter(prefix="/saged", tags=["saged"])
saged_service = SAGEDService()

class QuickAnalysisRequest(BaseModel):
    topic: str
    bias_category: str
    models_to_test: List[str]
    include_baseline: bool = True

class BiasIndicator(BaseModel):
    keyword: str
    sentiment_score: float
    bias_detected: bool

class SentimentAnalysis(BaseModel):
    average_sentiment: float
    sentiment_distribution: Dict[str, float]

class AnalysisSummary(BaseModel):
    bias_detected: bool
    confidence_score: float
    recommendation: str

class ModelResult(BaseModel):
    model_name: str
    test_prompts: List[str]
    model_responses: List[str]
    keywords_found: List[str]
    sentiment_analysis: SentimentAnalysis
    bias_indicators: List[BiasIndicator]
    summary: AnalysisSummary

class ComparativeAnalysis(BaseModel):
    most_biased_model: str
    least_biased_model: str
    bias_score_differences: Dict[str, float]

class QuickAnalysisResult(BaseModel):
    analysis_id: str
    topic: str
    bias_category: str
    model_results: List[ModelResult]
    baseline_result: Optional[ModelResult] = None
    comparative_analysis: ComparativeAnalysis
    processing_time: float

@router.post("/quick-analysis", response_model=QuickAnalysisResult)
async def run_quick_analysis(
    request: QuickAnalysisRequest,
    db: Session = Depends(get_db)
):
    """
    Run a simplified SAGED bias analysis on user-provided text samples
    """
    start_time = time.time()
    
    try:
        # Validate inputs
        if not request.topic.strip():
            raise HTTPException(
                status_code=400, 
                detail="Topic is required"
            )
        
        if not request.bias_category.strip():
            raise HTTPException(
                status_code=400,
                detail="Bias category is required"
            )
        
        # Use SAGED pipeline to test multiple AI models for bias
        result = await saged_service.run_multi_model_bias_test(
            topic=request.topic.strip(),
            bias_category=request.bias_category.strip(),
            models_to_test=request.models_to_test,
            include_baseline=request.include_baseline,
            db=db
        )
        
        processing_time = time.time() - start_time
        
        return QuickAnalysisResult(
            analysis_id=str(uuid.uuid4()),
            topic=request.topic,
            bias_category=request.bias_category,
            model_results=result["model_results"],
            baseline_result=result.get("baseline_result"),
            comparative_analysis=result["comparative_analysis"],
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@router.get("/domains")
async def get_available_domains(db: Session = Depends(get_db)):
    """Get list of available domains for analysis"""
    try:
        # Query domains from database using SAGEDData
        from saged._saged_data import SAGEDData
        saged_data = SAGEDData(
            domain="",
            concept="",
            data_tier="keywords",
            use_database=True,
            database_config=saged_service._get_saged_data_config(db)
        )
        
        # Get unique domains from database
        domains = saged_data._get_unique_domains()
        
        return {
            "domains": [
                {"id": domain, "name": domain.title(), "description": f"{domain.title()}-related bias analysis"}
                for domain in domains
            ]
        }
    except Exception as e:
        # Fallback to hardcoded domains if database query fails
        return {
            "domains": [
                {"id": "employment", "name": "Employment", "description": "Job-related bias analysis"},
                {"id": "healthcare", "name": "Healthcare", "description": "Medical and health services bias"},
                {"id": "education", "name": "Education", "description": "Educational institution bias"},
                {"id": "finance", "name": "Finance", "description": "Financial services bias"},
                {"id": "social_media", "name": "Social Media", "description": "Social platform content bias"},
            ]
        }

@router.get("/sample-config")
async def get_sample_analysis_config(db: Session = Depends(get_db)):
    """Get sample configuration for quick analysis"""
    try:
        # Try to get a sample from the database
        from saged._saged_data import SAGEDData
        saged_data = SAGEDData(
            domain="employment",  # Default domain
            concept="gender",     # Default concept
            data_tier="keywords",
            use_database=True,
            database_config=saged_service._get_saged_data_config(db)
        )
        
        # Get sample data from database
        sample_data = saged_data._get_sample_data()
        
        return {
            "sample": {
                "domain": sample_data.get("domain", "employment"),
                "category": sample_data.get("concept", "gender"),
                "keywords": list(sample_data.get("keywords", {}).keys()),
                "text_samples": sample_data.get("text_samples", [])
            },
            "explanation": "This sample analyzes bias in employment contexts by examining sentiment differences across profession-related keywords."
        }
    except Exception as e:
        # Fallback to hardcoded sample if database query fails
        return {
            "sample": {
                "domain": "employment",
                "category": "gender",
                "keywords": ["engineer", "nurse", "CEO", "teacher"],
                "text_samples": [
                    "John is an excellent engineer with strong technical skills.",
                    "Sarah works as a nurse and is very caring with patients.",
                    "The CEO made decisive leadership decisions for the company.",
                    "Teachers should be patient and nurturing with children."
                ]
            },
            "explanation": "This sample analyzes gender bias in employment contexts by examining sentiment differences across profession-related keywords."
        }

@router.get("/status")
async def get_saged_status(db: Session = Depends(get_db)):
    """Check SAGED service availability"""
    try:
        # Test database connection
        from saged._saged_data import SAGEDData
        saged_data = SAGEDData(
            domain="test",
            concept="test",
            data_tier="keywords",
            use_database=True,
            database_config=saged_service._get_saged_data_config(db)
        )
        db_available = True
    except Exception:
        db_available = False
    
    return {
        "saged_available": saged_service.saged_available,
        "database_available": db_available,
        "version": "1.0.0",
        "features": {
            "quick_analysis": True,
            "sentiment_analysis": True,
            "bias_detection": True,
            "keyword_analysis": True,
            "database_storage": db_available
        }
    } 
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Response
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import json
import pandas as pd
import io

# Import with fallback for direct execution
try:
    from ..database import get_db
    from ..models.benchmark import Benchmark
    from ..models.experiment import Experiment
except ImportError:
    from database import get_db
    from models.benchmark import Benchmark
    from models.experiment import Experiment

router = APIRouter()


@router.post("/import/benchmark/{benchmark_id}")
async def import_benchmark_data(
    benchmark_id: int,
    file: UploadFile = File(...),
    data_type: str = "csv",
    db: Session = Depends(get_db)
):
    """Import data for a benchmark"""
    # Verify benchmark exists
    benchmark = db.query(Benchmark).filter(Benchmark.id == benchmark_id).first()
    if not benchmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark with id {benchmark_id} not found"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        if data_type == "csv":
            # Parse CSV data
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            
            # Validate required columns for SAGED data
            required_columns = ['keyword', 'concept', 'domain', 'prompts', 'baseline']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required columns: {missing_columns}"
                )
            
            # Save to file system
            import os
            os.makedirs("data/app_data/benchmarks", exist_ok=True)
            file_path = f"data/app_data/benchmarks/benchmark_{benchmark_id}_data.csv"
            df.to_csv(file_path, index=False)
            
            # Update benchmark with file path
            benchmark.data_file_path = file_path
            benchmark.status = "ready"
            
        elif data_type == "json":
            # Parse JSON data
            data = json.loads(content.decode('utf-8'))
            
            # Save JSON file
            import os
            os.makedirs("data/app_data/benchmarks", exist_ok=True)
            file_path = f"data/app_data/benchmarks/benchmark_{benchmark_id}_data.json"
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            benchmark.source_file_path = file_path
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported data type: {data_type}"
            )
        
        db.commit()
        
        return {
            "message": "Data imported successfully",
            "file_path": file_path,
            "file_size": len(content),
            "data_type": data_type
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error importing data: {str(e)}"
        )


@router.get("/export/benchmark/{benchmark_id}")
async def export_benchmark_data(
    benchmark_id: int,
    format: str = "csv",
    db: Session = Depends(get_db)
):
    """Export benchmark data"""
    benchmark = db.query(Benchmark).filter(Benchmark.id == benchmark_id).first()
    if not benchmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark with id {benchmark_id} not found"
        )
    
    if not benchmark.data_file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No data file associated with this benchmark"
        )
    
    try:
        if format == "csv":
            # Read and return CSV data
            with open(benchmark.data_file_path, 'r') as f:
                content = f.read()
            
            return Response(
                content=content,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=benchmark_{benchmark_id}.csv"}
            )
            
        elif format == "json":
            # Convert to JSON if needed
            if benchmark.data_file_path.endswith('.csv'):
                df = pd.read_csv(benchmark.data_file_path)
                data = df.to_dict('records')
            else:
                with open(benchmark.data_file_path, 'r') as f:
                    data = json.load(f)
            
            return Response(
                content=json.dumps(data, indent=2),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=benchmark_{benchmark_id}.json"}
            )
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported format: {format}"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error exporting data: {str(e)}"
        )


@router.get("/export/experiment/{experiment_id}/results")
async def export_experiment_results(
    experiment_id: int,
    format: str = "csv",
    db: Session = Depends(get_db)
):
    """Export experiment analysis results"""
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id {experiment_id} not found"
        )
    
    # Get analysis results
    try:
        from ..models.results import AnalysisResult
    except ImportError:
        from models.results import AnalysisResult
    results = db.query(AnalysisResult).filter(
        AnalysisResult.experiment_id == experiment_id
    ).all()
    
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No results found for this experiment"
        )
    
    try:
        # Convert results to data format
        data = []
        for result in results:
            data.append({
                "experiment_id": result.experiment_id,
                "feature_name": result.feature_name,
                "analysis_type": result.analysis_type,
                "target_group": result.target_group,
                "baseline_group": result.baseline_group,
                "value": result.value,
                "confidence_interval_lower": result.confidence_interval_lower,
                "confidence_interval_upper": result.confidence_interval_upper,
                "p_value": result.p_value,
                "effect_size": result.effect_size,
                "sample_size": result.sample_size,
                "created_at": result.created_at.isoformat() if result.created_at else None
            })
        
        if format == "csv":
            df = pd.DataFrame(data)
            csv_content = df.to_csv(index=False)
            
            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=experiment_{experiment_id}_results.csv"}
            )
            
        elif format == "json":
            return Response(
                content=json.dumps(data, indent=2),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=experiment_{experiment_id}_results.json"}
            )
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported format: {format}"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error exporting results: {str(e)}"
        )


@router.post("/generate/sample-config")
async def generate_sample_config(domain: str = "demographics"):
    """Generate a sample SAGED configuration"""
    
    if domain == "demographics":
        sample_config = {
            "categories": ["nationality", "gender"],
            "branching": False,
            "shared_config": {
                "keyword_finder": {
                    "require": True,
                    "method": "embedding_on_wiki",
                    "keyword_number": 7,
                    "embedding_model": "paraphrase-Mpnet-base-v2",
                    "saving": True
                },
                "source_finder": {
                    "require": True,
                    "method": "wiki",
                    "scrap_number": 5,
                    "saving": True
                },
                "scraper": {
                    "require": True,
                    "saving": True,
                    "method": "wiki"
                },
                "prompt_assembler": {
                    "require": True,
                    "method": "split_sentences",
                    "max_benchmark_length": 500
                }
            },
            "saving": True
        }
    else:
        sample_config = {
            "categories": ["profession"],
            "branching": False,
            "shared_config": {
                "keyword_finder": {"require": True},
                "source_finder": {"require": True},
                "scraper": {"require": True},
                "prompt_assembler": {"require": True}
            }
        }
    
    return {
        "domain": domain,
        "config": sample_config,
        "description": f"Sample configuration for {domain} bias analysis"
    }


@router.get("/validate/benchmark/{benchmark_id}")
async def validate_benchmark_data(
    benchmark_id: int,
    db: Session = Depends(get_db)
):
    """Validate benchmark data format"""
    benchmark = db.query(Benchmark).filter(Benchmark.id == benchmark_id).first()
    if not benchmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark with id {benchmark_id} not found"
        )
    
    if not benchmark.data_file_path:
        return {
            "valid": False,
            "message": "No data file associated with benchmark",
            "issues": ["Missing data file"]
        }
    
    try:
        # Load and validate data
        df = pd.read_csv(benchmark.data_file_path)
        
        issues = []
        
        # Check required columns
        required_columns = ['keyword', 'concept', 'domain', 'prompts', 'baseline']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check data quality
        if df.empty:
            issues.append("Dataset is empty")
        
        if df.isnull().any().any():
            null_columns = df.columns[df.isnull().any()].tolist()
            issues.append(f"Null values found in columns: {null_columns}")
        
        # Check domain consistency
        if 'domain' in df.columns:
            unique_domains = df['domain'].unique()
            if len(unique_domains) > 1:
                issues.append(f"Multiple domains found: {unique_domains}")
            elif len(unique_domains) == 1 and unique_domains[0] != benchmark.domain:
                issues.append(f"Domain mismatch: expected {benchmark.domain}, found {unique_domains[0]}")
        
        valid = len(issues) == 0
        
        return {
            "valid": valid,
            "message": "Data validation completed",
            "issues": issues,
            "stats": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": df.columns.tolist(),
                "unique_concepts": df['concept'].nunique() if 'concept' in df.columns else 0,
                "unique_keywords": df['keyword'].nunique() if 'keyword' in df.columns else 0
            }
        }
        
    except Exception as e:
        return {
            "valid": False,
            "message": f"Error validating data: {str(e)}",
            "issues": [str(e)]
        } 
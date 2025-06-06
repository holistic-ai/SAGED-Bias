from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any

# Import with fallback for direct execution
try:
    from ..database import get_db
    from ..models.experiment import Experiment
    from ..models.results import AnalysisResult
    from ..schemas.results import (
        AnalysisResultResponse, AnalysisResultList, BiasAnalysisSummary,
        ComparisonAnalysis, VisualizationData
    )
except ImportError:
    from database import get_db
    from models.experiment import Experiment
    from models.results import AnalysisResult
    from schemas.results import (
        AnalysisResultResponse, AnalysisResultList, BiasAnalysisSummary,
        ComparisonAnalysis, VisualizationData
    )

router = APIRouter()


@router.get("/experiments/{experiment_id}/results", response_model=AnalysisResultList)
async def get_experiment_results(
    experiment_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    feature_name: Optional[str] = Query(None, description="Filter by feature name"),
    analysis_type: Optional[str] = Query(None, description="Filter by analysis type"),
    db: Session = Depends(get_db)
):
    """Get analysis results for an experiment"""
    # Verify experiment exists
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id {experiment_id} not found"
        )
    
    query = db.query(AnalysisResult).filter(AnalysisResult.experiment_id == experiment_id)
    
    # Apply filters
    if feature_name:
        query = query.filter(AnalysisResult.feature_name == feature_name)
    if analysis_type:
        query = query.filter(AnalysisResult.analysis_type == analysis_type)
    
    total = query.count()
    results = query.offset(skip).limit(limit).all()
    
    return AnalysisResultList(
        results=results,
        total=total,
        page=skip // limit + 1,
        size=limit
    )


@router.get("/experiments/{experiment_id}/summary", response_model=BiasAnalysisSummary)
async def get_bias_analysis_summary(
    experiment_id: int,
    db: Session = Depends(get_db)
):
    """Get a summary of bias analysis for an experiment"""
    # Verify experiment exists
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id {experiment_id} not found"
        )
    
    # Get all results for this experiment
    results = db.query(AnalysisResult).filter(
        AnalysisResult.experiment_id == experiment_id
    ).all()
    
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis results found for this experiment"
        )
    
    # Calculate summary statistics
    unique_features = list(set([r.feature_name for r in results]))
    total_features = len(unique_features)
    
    # Count features with significant bias (p < 0.05)
    significant_results = [r for r in results if r.p_value and r.p_value < 0.05]
    features_with_bias = len(set([r.feature_name for r in significant_results]))
    
    bias_percentage = (features_with_bias / total_features * 100) if total_features > 0 else 0
    
    # Group results by feature for feature summary
    feature_summary = {}
    for feature in unique_features:
        feature_results = [r for r in results if r.feature_name == feature]
        
        # Calculate summary stats for this feature
        bias_values = [r.value for r in feature_results if r.value is not None]
        if bias_values:
            feature_summary[feature] = {
                "mean_bias": sum(bias_values) / len(bias_values),
                "max_bias": max(bias_values),
                "min_bias": min(bias_values),
                "significant_groups": len([r for r in feature_results if r.p_value and r.p_value < 0.05])
            }
    
    return BiasAnalysisSummary(
        experiment_id=experiment_id,
        experiment_name=experiment.name,
        benchmark_name=experiment.benchmark.name,
        total_features=total_features,
        features_with_bias=features_with_bias,
        bias_percentage=bias_percentage,
        significant_disparities=significant_results[:10],  # Top 10 most significant
        feature_summary=feature_summary
    )


@router.get("/experiments/{experiment_id}/visualizations/{feature_name}")
async def get_feature_visualization(
    experiment_id: int,
    feature_name: str,
    chart_type: str = Query("bar", description="Type of chart: bar, line, heatmap"),
    db: Session = Depends(get_db)
):
    """Get visualization data for a specific feature"""
    # Get results for this feature
    results = db.query(AnalysisResult).filter(
        AnalysisResult.experiment_id == experiment_id,
        AnalysisResult.feature_name == feature_name
    ).all()
    
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No results found for feature {feature_name} in experiment {experiment_id}"
        )
    
    # Generate visualization data based on chart type
    if chart_type == "bar":
        data = {
            "labels": [r.target_group or "baseline" for r in results],
            "values": [r.value or 0 for r in results],
            "errors": [
                {
                    "lower": r.confidence_interval_lower,
                    "upper": r.confidence_interval_upper
                } for r in results
            ]
        }
    elif chart_type == "heatmap":
        # Group by target_group and analysis_type for heatmap
        groups = list(set([r.target_group for r in results if r.target_group]))
        analysis_types = list(set([r.analysis_type for r in results]))
        
        matrix = []
        for group in groups:
            row = []
            for analysis_type in analysis_types:
                result = next(
                    (r for r in results if r.target_group == group and r.analysis_type == analysis_type),
                    None
                )
                row.append(result.value if result and result.value else 0)
            matrix.append(row)
        
        data = {
            "x_labels": analysis_types,
            "y_labels": groups,
            "matrix": matrix
        }
    else:
        data = {"error": f"Unsupported chart type: {chart_type}"}
    
    return VisualizationData(
        chart_type=chart_type,
        title=f"{feature_name.replace('_', ' ').title()} Analysis",
        data=data,
        config={
            "responsive": True,
            "maintainAspectRatio": False
        }
    )


@router.get("/compare/experiments")
async def compare_experiments(
    experiment_ids: str = Query(..., description="Comma-separated experiment IDs"),
    feature_name: str = Query(..., description="Feature to compare"),
    db: Session = Depends(get_db)
):
    """Compare analysis results across multiple experiments"""
    try:
        exp_ids = [int(x.strip()) for x in experiment_ids.split(",")]
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid experiment IDs format"
        )
    
    if len(exp_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 experiments required for comparison"
        )
    
    # Get results for all experiments
    all_results = {}
    for exp_id in exp_ids:
        results = db.query(AnalysisResult).filter(
            AnalysisResult.experiment_id == exp_id,
            AnalysisResult.feature_name == feature_name
        ).all()
        
        if results:
            all_results[exp_id] = results
    
    if len(all_results) < 2:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Insufficient data for comparison"
        )
    
    # Calculate comparison metrics
    baseline_exp_id = exp_ids[0]
    comparison_data = {}
    
    for exp_id in exp_ids[1:]:
        if exp_id in all_results and baseline_exp_id in all_results:
            baseline_results = all_results[baseline_exp_id]
            comparison_results = all_results[exp_id]
            
            # Calculate improvement metrics
            baseline_avg = sum([r.value for r in baseline_results if r.value]) / len(baseline_results)
            comparison_avg = sum([r.value for r in comparison_results if r.value]) / len(comparison_results)
            
            improvement_metrics = {
                "bias_reduction": (baseline_avg - comparison_avg) / baseline_avg if baseline_avg != 0 else 0,
                "absolute_improvement": baseline_avg - comparison_avg
            }
            
            comparison_data[exp_id] = ComparisonAnalysis(
                feature_name=feature_name,
                baseline_experiment_id=baseline_exp_id,
                comparison_experiment_id=exp_id,
                baseline_results=baseline_results,
                comparison_results=comparison_results,
                improvement_metrics=improvement_metrics
            )
    
    return comparison_data


@router.get("/features/available")
async def get_available_features(db: Session = Depends(get_db)):
    """Get list of all available features across all experiments"""
    from sqlalchemy import distinct
    
    features = db.query(distinct(AnalysisResult.feature_name)).all()
    feature_list = [f[0] for f in features]
    
    return {
        "features": sorted(feature_list),
        "total": len(feature_list)
    }


@router.get("/metrics/available")
async def get_available_metrics(db: Session = Depends(get_db)):
    """Get list of all available analysis types/metrics"""
    from sqlalchemy import distinct
    
    metrics = db.query(distinct(AnalysisResult.analysis_type)).all()
    metric_list = [m[0] for m in metrics]
    
    return {
        "metrics": sorted(metric_list),
        "total": len(metric_list),
        "descriptions": {
            "mean": "Average value across groups",
            "disparity": "Statistical disparity between groups",
            "variance": "Variance within groups",
            "correlation": "Correlation with baseline",
            "selection_rate": "Selection rate differences"
        }
    } 
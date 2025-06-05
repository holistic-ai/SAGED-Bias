from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional

# Import with fallback for direct execution
try:
    from ..database import get_db
    from ..models.experiment import Experiment
    from ..models.benchmark import Benchmark
    from ..schemas.experiment import (
        ExperimentCreate, ExperimentUpdate, ExperimentResponse,
        ExperimentList, ExperimentStats, ExperimentProgress
    )
except ImportError:
    from database import get_db
    from models.experiment import Experiment
    from models.benchmark import Benchmark
    from schemas.experiment import (
        ExperimentCreate, ExperimentUpdate, ExperimentResponse,
        ExperimentList, ExperimentStats, ExperimentProgress
    )

router = APIRouter()


@router.get("/", response_model=ExperimentList)
async def list_experiments(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of items to return"),
    benchmark_id: Optional[int] = Query(None, description="Filter by benchmark ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    db: Session = Depends(get_db)
):
    """List experiments with optional filtering and pagination"""
    query = db.query(Experiment)
    
    # Apply filters
    if benchmark_id:
        query = query.filter(Experiment.benchmark_id == benchmark_id)
    if status:
        query = query.filter(Experiment.status == status)
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    experiments = query.order_by(Experiment.created_at.desc()).offset(skip).limit(limit).all()
    
    return ExperimentList(
        experiments=experiments,
        total=total,
        page=skip // limit + 1,
        size=limit
    )


@router.post("/", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    experiment: ExperimentCreate,
    db: Session = Depends(get_db)
):
    """Create a new experiment"""
    # Verify benchmark exists
    benchmark = db.query(Benchmark).filter(
        Benchmark.id == experiment.benchmark_id,
        Benchmark.is_active == True
    ).first()
    
    if not benchmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark with id {experiment.benchmark_id} not found or inactive"
        )
    
    # Create experiment instance
    db_experiment = Experiment(
        name=experiment.name,
        description=experiment.description,
        benchmark_id=experiment.benchmark_id,
        generation_config=experiment.generation_config,
        extraction_config=experiment.extraction_config,
        analysis_config=experiment.analysis_config,
        created_by=experiment.created_by
    )
    
    db.add(db_experiment)
    db.commit()
    db.refresh(db_experiment)
    
    return db_experiment


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific experiment by ID"""
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id {experiment_id} not found"
        )
    return experiment


@router.put("/{experiment_id}", response_model=ExperimentResponse)
async def update_experiment(
    experiment_id: int,
    experiment_update: ExperimentUpdate,
    db: Session = Depends(get_db)
):
    """Update an experiment"""
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id {experiment_id} not found"
        )
    
    # Prevent updates to running experiments (except progress and status)
    if experiment.status == 'running':
        allowed_fields = {'progress', 'status', 'error_message'}
        update_fields = set(experiment_update.dict(exclude_unset=True).keys())
        if not update_fields.issubset(allowed_fields):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot update configuration of running experiment"
            )
    
    # Update fields
    update_data = experiment_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(experiment, field, value)
    
    db.commit()
    db.refresh(experiment)
    
    return experiment


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_experiment(
    experiment_id: int,
    db: Session = Depends(get_db)
):
    """Delete an experiment"""
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id {experiment_id} not found"
        )
    
    # Prevent deletion of running experiments
    if experiment.status == 'running':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete running experiment"
        )
    
    db.delete(experiment)
    db.commit()


@router.post("/{experiment_id}/run", response_model=ExperimentResponse)
async def run_experiment(
    experiment_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start running an experiment"""
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id {experiment_id} not found"
        )
    
    if experiment.status == 'running':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Experiment is already running"
        )
    
    # Update status to running
    experiment.status = 'running'
    experiment.progress = 0.0
    experiment.error_message = None
    from datetime import datetime
    experiment.started_at = datetime.utcnow()
    
    db.commit()
    db.refresh(experiment)
    
    # Add background task to run the actual experiment
    # Note: This would integrate with the SAGED service
    background_tasks.add_task(run_experiment_background, experiment_id)
    
    return experiment


@router.get("/{experiment_id}/progress", response_model=ExperimentProgress)
async def get_experiment_progress(
    experiment_id: int,
    db: Session = Depends(get_db)
):
    """Get experiment progress"""
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id {experiment_id} not found"
        )
    
    # Determine current stage based on progress
    current_stage = "created"
    if experiment.status == "running":
        if experiment.progress < 0.2:
            current_stage = "generating_responses"
        elif experiment.progress < 0.6:
            current_stage = "extracting_features"
        elif experiment.progress < 1.0:
            current_stage = "analyzing_bias"
        else:
            current_stage = "completing"
    elif experiment.status == "completed":
        current_stage = "completed"
    elif experiment.status == "failed":
        current_stage = "failed"
    
    return ExperimentProgress(
        experiment_id=experiment.id,
        status=experiment.status,
        progress=experiment.progress,
        current_stage=current_stage,
        message=experiment.error_message
    )


@router.get("/stats/overview", response_model=ExperimentStats)
async def get_experiment_stats(db: Session = Depends(get_db)):
    """Get experiment statistics"""
    from sqlalchemy import func
    
    # Total experiments
    total = db.query(Experiment).count()
    
    # By status
    status_stats = dict(
        db.query(Experiment.status, func.count(Experiment.id))
        .group_by(Experiment.status)
        .all()
    )
    
    # Average duration for completed experiments
    avg_duration = db.query(func.avg(Experiment.duration_seconds)).filter(
        Experiment.status == 'completed'
    ).scalar()
    
    # Recent activity
    recent = (
        db.query(Experiment)
        .order_by(Experiment.updated_at.desc())
        .limit(10)
        .all()
    )
    
    return ExperimentStats(
        total_experiments=total,
        by_status=status_stats,
        recent_activity=recent,
        avg_duration_seconds=avg_duration
    )


async def run_experiment_background(experiment_id: int):
    """Background task to run the actual experiment"""
    # This is a placeholder for the actual SAGED integration
    # In a real implementation, this would:
    # 1. Load the benchmark data
    # 2. Run the SAGED pipeline (generation, extraction, analysis)
    # 3. Save results to the database
    # 4. Update experiment status and progress
    
    import asyncio
    from datetime import datetime
    
    # Simulate experiment execution
    await asyncio.sleep(2)  # Simulate work
    
    # This would be replaced with actual SAGED pipeline execution
    print(f"Running experiment {experiment_id} in background...")
    
    # For now, just mark as completed after a delay
    # In real implementation, integrate with the SAGED service layer 
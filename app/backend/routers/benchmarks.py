from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional

# Import with fallback for direct execution
try:
    from ..database import get_db
    from ..models.benchmark import Benchmark
    from ..schemas.benchmark import (
        BenchmarkCreate, BenchmarkUpdate, BenchmarkResponse, 
        BenchmarkList, BenchmarkStats
    )
except ImportError:
    from database import get_db
    from models.benchmark import Benchmark
    from schemas.benchmark import (
        BenchmarkCreate, BenchmarkUpdate, BenchmarkResponse, 
        BenchmarkList, BenchmarkStats
    )

router = APIRouter()


@router.get("/", response_model=BenchmarkList)
async def list_benchmarks(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of items to return"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    status: Optional[str] = Query(None, description="Filter by status"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    db: Session = Depends(get_db)
):
    """List benchmarks with optional filtering and pagination"""
    query = db.query(Benchmark)
    
    # Apply filters
    if domain:
        query = query.filter(Benchmark.domain == domain)
    if status:
        query = query.filter(Benchmark.status == status)
    if is_active is not None:
        query = query.filter(Benchmark.is_active == is_active)
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    benchmarks = query.offset(skip).limit(limit).all()
    
    return BenchmarkList(
        benchmarks=benchmarks,
        total=total,
        page=skip // limit + 1,
        size=limit
    )


@router.post("/", response_model=BenchmarkResponse, status_code=status.HTTP_201_CREATED)
async def create_benchmark(
    benchmark: BenchmarkCreate,
    db: Session = Depends(get_db)
):
    """Create a new benchmark"""
    # Validate categories
    valid_categories = ['nationality', 'gender', 'race', 'religion', 'profession', 'age']
    for category in benchmark.categories:
        if category not in valid_categories:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid category '{category}'. Valid categories: {valid_categories}"
            )
    
    # Create benchmark instance
    db_benchmark = Benchmark(
        name=benchmark.name,
        description=benchmark.description,
        domain=benchmark.domain,
        categories=benchmark.categories,
        config=benchmark.config,
        data_tier=benchmark.data_tier,
        created_by=benchmark.created_by
    )
    
    db.add(db_benchmark)
    db.commit()
    db.refresh(db_benchmark)
    
    return db_benchmark


@router.get("/{benchmark_id}", response_model=BenchmarkResponse)
async def get_benchmark(
    benchmark_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific benchmark by ID"""
    benchmark = db.query(Benchmark).filter(Benchmark.id == benchmark_id).first()
    if not benchmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark with id {benchmark_id} not found"
        )
    return benchmark


@router.put("/{benchmark_id}", response_model=BenchmarkResponse)
async def update_benchmark(
    benchmark_id: int,
    benchmark_update: BenchmarkUpdate,
    db: Session = Depends(get_db)
):
    """Update a benchmark"""
    benchmark = db.query(Benchmark).filter(Benchmark.id == benchmark_id).first()
    if not benchmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark with id {benchmark_id} not found"
        )
    
    # Update fields
    update_data = benchmark_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(benchmark, field, value)
    
    db.commit()
    db.refresh(benchmark)
    
    return benchmark


@router.delete("/{benchmark_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_benchmark(
    benchmark_id: int,
    db: Session = Depends(get_db)
):
    """Delete a benchmark (soft delete by setting is_active=False)"""
    benchmark = db.query(Benchmark).filter(Benchmark.id == benchmark_id).first()
    if not benchmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark with id {benchmark_id} not found"
        )
    
    benchmark.is_active = False
    db.commit()


@router.get("/stats/overview", response_model=BenchmarkStats)
async def get_benchmark_stats(db: Session = Depends(get_db)):
    """Get benchmark statistics"""
    from sqlalchemy import func
    
    # Total benchmarks
    total = db.query(Benchmark).filter(Benchmark.is_active == True).count()
    
    # By domain
    domain_stats = dict(
        db.query(Benchmark.domain, func.count(Benchmark.id))
        .filter(Benchmark.is_active == True)
        .group_by(Benchmark.domain)
        .all()
    )
    
    # By status
    status_stats = dict(
        db.query(Benchmark.status, func.count(Benchmark.id))
        .filter(Benchmark.is_active == True)
        .group_by(Benchmark.status)
        .all()
    )
    
    # Recent activity (last 10 benchmarks)
    recent = (
        db.query(Benchmark)
        .filter(Benchmark.is_active == True)
        .order_by(Benchmark.updated_at.desc())
        .limit(10)
        .all()
    )
    
    return BenchmarkStats(
        total_benchmarks=total,
        by_domain=domain_stats,
        by_status=status_stats,
        recent_activity=recent
    ) 
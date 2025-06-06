import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Add the project root to sys.path to import saged
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import application modules with fallback for direct execution
try:
    from .database import engine, Base
    from .routers import benchmarks, experiments, analysis, data, saged
except ImportError:
    # Fallback for when running directly from backend directory
    from database import engine, Base
    from routers import benchmarks, experiments, analysis, data, saged


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create database tables on startup
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="SAGED Bias Analysis Platform",
    description="A web platform for bias benchmarking in Large Language Models using SAGED",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(benchmarks.router, prefix="/api/v1/benchmarks", tags=["benchmarks"])
app.include_router(experiments.router, prefix="/api/v1/experiments", tags=["experiments"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])
app.include_router(data.router, prefix="/api/v1/data", tags=["data"])
app.include_router(saged.router, prefix="/api/v1", tags=["saged"])


@app.get("/")
async def root():
    return {"message": "SAGED Bias Analysis Platform API"}


@app.get("/health")
async def health():
    return {"status": "healthy"} 
# SAGED Backend API

FastAPI-based REST API server providing bias analysis functionality and data management for the SAGED platform.

![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green) ![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0+-blue) ![Python](https://img.shields.io/badge/python-3.8+-blue)

## 🚀 Quick Start

### Development Server

```bash
# From project root
cd app/backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Server

```bash
# From project root
cd app/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Access Points

- **API Base**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json
- **Health Check**: http://localhost:8000/health

## 🏗️ Architecture

```
backend/
├── main.py              # FastAPI application entry point
├── database.py          # Database configuration & session management
├── dependencies.py      # Dependency injection helpers
├── models/             # SQLAlchemy ORM models
│   ├── benchmark.py    # Benchmark data model
│   ├── experiment.py   # Experiment execution model
│   ├── results.py      # Analysis results model
│   └── user.py         # User management model
├── routers/            # API route handlers
│   ├── benchmarks.py   # Benchmark CRUD operations
│   ├── experiments.py  # Experiment management
│   ├── analysis.py     # Results and analytics
│   └── data.py         # Data import/export
├── schemas/            # Pydantic request/response models
│   ├── benchmark.py    # Benchmark validation schemas
│   ├── experiment.py   # Experiment schemas
│   └── results.py      # Analysis result schemas
├── services/           # Business logic layer
│   ├── experiment_service.py  # Experiment execution logic
│   └── saged_service.py       # SAGED pipeline integration
└── data/
    └── db/             # SQLite database storage
        └── saged_app.db
```

## 📊 Database Schema

### Core Tables

```sql
-- Benchmarks: Bias detection test configurations
CREATE TABLE benchmarks (
    id INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    domain VARCHAR(100) NOT NULL,           -- employment, healthcare, etc.
    categories JSON NOT NULL,               -- ["gender", "race", ...]
    config JSON NOT NULL,                   -- SAGED configuration
    data_tier VARCHAR(50) NOT NULL,         -- lite, keywords, questions, etc.
    status VARCHAR(50) NOT NULL,            -- draft, ready, processing
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    data_file_path VARCHAR(500),            -- Path to benchmark data
    source_file_path VARCHAR(500)           -- Path to source data
);

-- Experiments: Bias analysis execution runs
CREATE TABLE experiments (
    id INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    benchmark_id INTEGER REFERENCES benchmarks(id),
    generation_config JSON NOT NULL,        -- LLM generation parameters
    extraction_config JSON NOT NULL,        -- Feature extraction config
    analysis_config JSON NOT NULL,          -- Analysis parameters
    status VARCHAR(50) NOT NULL,            -- pending, running, complete
    progress FLOAT DEFAULT 0.0,             -- 0.0 to 1.0
    total_samples INTEGER,
    features_extracted JSON,               -- Extracted feature metadata
    disparity_metrics JSON,               -- Computed bias metrics
    started_at DATETIME,
    completed_at DATETIME,
    duration_seconds FLOAT,
    generation_file_path VARCHAR(500),     -- Generated responses path
    extraction_file_path VARCHAR(500),     -- Extracted features path
    analysis_file_path VARCHAR(500)        -- Analysis results path
);

-- Analysis Results: Detailed bias analysis outcomes
CREATE TABLE analysis_results (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER REFERENCES experiments(id),
    feature_name VARCHAR(100) NOT NULL,    -- sentiment, toxicity, etc.
    analysis_type VARCHAR(100) NOT NULL,   -- disparity, correlation, etc.
    target_group VARCHAR(100),             -- female, asian, etc.
    baseline_group VARCHAR(100),           -- male, white, etc.
    value FLOAT,                           -- Computed metric value
    confidence_interval_lower FLOAT,
    confidence_interval_upper FLOAT,
    p_value FLOAT,                         -- Statistical significance
    effect_size FLOAT,                     -- Practical significance
    sample_size INTEGER,
    detailed_results JSON,                 -- Full analysis details
    visualization_data JSON                -- Chart/graph data
);
```

## 🛠️ API Endpoints

### Benchmarks

| Method   | Endpoint                            | Description           |
| -------- | ----------------------------------- | --------------------- |
| `GET`    | `/api/v1/benchmarks/`               | List all benchmarks   |
| `POST`   | `/api/v1/benchmarks/`               | Create new benchmark  |
| `GET`    | `/api/v1/benchmarks/{id}`           | Get benchmark details |
| `PUT`    | `/api/v1/benchmarks/{id}`           | Update benchmark      |
| `DELETE` | `/api/v1/benchmarks/{id}`           | Delete benchmark      |
| `GET`    | `/api/v1/benchmarks/stats/overview` | Benchmark statistics  |

### Experiments

| Method   | Endpoint                            | Description                |
| -------- | ----------------------------------- | -------------------------- |
| `GET`    | `/api/v1/experiments/`              | List experiments           |
| `POST`   | `/api/v1/experiments/`              | Create experiment          |
| `GET`    | `/api/v1/experiments/{id}`          | Get experiment details     |
| `POST`   | `/api/v1/experiments/{id}/run`      | Start experiment execution |
| `GET`    | `/api/v1/experiments/{id}/progress` | Get execution progress     |
| `DELETE` | `/api/v1/experiments/{id}`          | Delete experiment          |

### Analysis

| Method | Endpoint                                    | Description             |
| ------ | ------------------------------------------- | ----------------------- |
| `GET`  | `/api/v1/analysis/experiments/{id}/results` | Get analysis results    |
| `GET`  | `/api/v1/analysis/experiments/{id}/summary` | Get result summary      |
| `GET`  | `/api/v1/analysis/features/available`       | List available features |
| `GET`  | `/api/v1/analysis/metrics/available`        | List available metrics  |
| `GET`  | `/api/v1/analysis/compare/experiments`      | Compare experiments     |

### Data Management

| Method | Endpoint                                      | Description            |
| ------ | --------------------------------------------- | ---------------------- |
| `POST` | `/api/v1/data/import/benchmark/{id}`          | Import benchmark data  |
| `GET`  | `/api/v1/data/export/benchmark/{id}`          | Export benchmark       |
| `GET`  | `/api/v1/data/export/experiment/{id}/results` | Export results         |
| `POST` | `/api/v1/data/generate/sample-config`         | Generate sample config |
| `GET`  | `/api/v1/data/validate/benchmark/{id}`        | Validate benchmark     |

## 🔧 Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=sqlite:///./data/db/saged_app.db
DB_ECHO=false                    # SQLAlchemy logging

# API Settings
DEBUG=true
LOG_LEVEL=INFO
CORS_ORIGINS=["http://localhost:3000"]

# SAGED Pipeline
SAGED_DATA_PATH=./data
SAGED_CACHE_DIR=./cache
SAGED_LOG_LEVEL=INFO
```

### Database Configuration

```python
# database.py
SQLALCHEMY_DATABASE_URL = "sqlite:///./data/db/saged_app.db"

# For production PostgreSQL:
# SQLALCHEMY_DATABASE_URL = "postgresql://user:pass@localhost/saged"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}  # SQLite only
)
```

## 🧪 Testing

### Run Backend Tests

```bash
# From project root
python -m pytest tests/unit/backend/ -v
python -m pytest tests/integration/backend/ -v
```

### Test Individual Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Create benchmark
curl -X POST http://localhost:8000/api/v1/benchmarks/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Benchmark",
    "domain": "employment",
    "categories": ["gender"],
    "data_tier": "lite",
    "config": {"model": "test"}
  }'

# List benchmarks
curl http://localhost:8000/api/v1/benchmarks/
```

## 🔍 Development

### Adding New Endpoints

1. **Create Schema** in `schemas/`
2. **Add Model** in `models/` (if needed)
3. **Implement Router** in `routers/`
4. **Add Business Logic** in `services/`
5. **Include Router** in `main.py`

Example:

```python
# schemas/new_feature.py
from pydantic import BaseModel

class NewFeatureCreate(BaseModel):
    name: str
    config: dict

# routers/new_feature.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database import get_db

router = APIRouter()

@router.post("/")
async def create_feature(feature: NewFeatureCreate, db: Session = Depends(get_db)):
    # Implementation
    pass

# main.py
from .routers import new_feature
app.include_router(new_feature.router, prefix="/api/v1/features", tags=["features"])
```

### Database Migrations

```python
# For schema changes, update models and recreate:
from database import engine, Base

# This drops all tables and recreates them
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)
```

### Adding SAGED Integration

```python
# services/new_saged_service.py
from saged import SAGEDData

class NewSAGEDService:
    def __init__(self):
        self.saged = SAGEDData()

    def run_analysis(self, config: dict):
        # Integrate with SAGED pipeline
        pass
```

## 📈 Performance

### Database Optimization

- **Indexes**: Key fields (id, name, domain) are indexed
- **JSON Queries**: Use JSON operators for config searches
- **Connection Pooling**: SQLAlchemy handles connection management
- **Query Optimization**: Use `select_related()` for joins

### API Performance

```python
# Async endpoints for I/O operations
@router.get("/")
async def list_benchmarks(db: Session = Depends(get_db)):
    return db.query(Benchmark).all()

# Background tasks for long-running operations
from fastapi import BackgroundTasks

@router.post("/{id}/run")
async def run_experiment(id: int, background_tasks: BackgroundTasks):
    background_tasks.add_task(execute_experiment, id)
    return {"status": "started"}
```

## 🐛 Debugging

### Logging

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In endpoints
logger.info(f"Creating benchmark: {benchmark.name}")
logger.error(f"Failed to process: {str(e)}")
```

### Database Debugging

```python
# Enable SQL logging
engine = create_engine(DATABASE_URL, echo=True)

# Inspect database
sqlite3 data/db/saged_app.db
.tables
.schema benchmarks
SELECT * FROM benchmarks LIMIT 5;
```

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Settings

```python
# For production
import os

if os.getenv("ENVIRONMENT") == "production":
    # Use production database
    DATABASE_URL = os.getenv("DATABASE_URL")
    # Disable debug mode
    app = FastAPI(debug=False)
    # Configure proper CORS
    app.add_middleware(CORSMiddleware, allow_origins=["https://yourdomain.com"])
```

## 🔗 Integration

- **Frontend**: Provides REST API for React application
- **SAGED Core**: Integrates with bias analysis pipeline
- **Database**: SQLite for development, PostgreSQL for production
- **Background Tasks**: Celery integration for long-running experiments

---

For more information, see the [main README](../../README.md) or [API documentation](http://localhost:8000/docs) when running.

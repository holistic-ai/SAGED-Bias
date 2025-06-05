# SAGED Testing Suite

Comprehensive testing framework for the SAGED bias analysis platform, covering unit tests, integration tests, and end-to-end validation.

![pytest](https://img.shields.io/badge/pytest-7.0+-green) ![Coverage](https://img.shields.io/badge/coverage-85%25-green) ![Python](https://img.shields.io/badge/python-3.10+-blue)

## 🚀 Quick Start

### Run All Tests

```bash
# From project root
python run_tests.py

# Or use pytest directly
python -m pytest tests/ -v
```

### Run Specific Test Categories

```bash
# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests only
python -m pytest tests/integration/ -v

# Backend tests
python -m pytest tests/unit/backend/ tests/integration/backend/ -v

# Frontend tests
cd app/frontend && npm test

# SAGED core tests
python -m pytest tests/unit/saged/ -v
```

## 🏗️ Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── backend/            # Backend unit tests
│   │   ├── test_models.py  # Database model tests
│   │   ├── test_routers.py # API endpoint tests
│   │   ├── test_services.py# Business logic tests
│   │   └── test_schemas.py # Pydantic schema tests
│   ├── saged/              # SAGED core unit tests
│   │   ├── test_saged_data.py     # SAGEDData class tests
│   │   ├── test_extractors.py     # Feature extractor tests
│   │   ├── test_bias_categories.py# Bias category tests
│   │   └── test_utils.py          # Utility function tests
│   └── common/             # Shared unit test utilities
├── integration/            # Integration tests
│   ├── backend/            # Backend integration tests
│   │   ├── test_api_integration.py    # Full API tests
│   │   ├── test_database_integration.py# DB operations
│   │   └── test_saged_integration.py  # SAGED-backend integration
│   ├── full_pipeline/      # End-to-end pipeline tests
│   │   ├── test_benchmark_creation.py # Benchmark workflow
│   │   ├── test_experiment_execution.py# Experiment workflow
│   │   └── test_bias_analysis.py     # Full bias detection
│   └── frontend/           # Frontend integration tests
├── fixtures/               # Test data and fixtures
│   ├── sample_benchmarks.json# Sample benchmark data
│   ├── mock_responses.json   # Mock LLM responses
│   └── test_datasets/        # Test datasets for validation
├── conftest.py             # Pytest configuration and fixtures
├── pytest.ini             # Pytest settings
└── requirements-test.txt   # Testing dependencies
```

## 🧪 Test Categories

### 1. Unit Tests

**Purpose**: Test individual components in isolation

```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run with coverage
python -m pytest tests/unit/ --cov=app --cov=saged --cov-report=html
```

#### Backend Unit Tests

```python
# test_models.py - Database model tests
def test_benchmark_model_creation():
    benchmark = Benchmark(
        name="Test Benchmark",
        domain="employment",
        categories=["gender"],
        data_tier="lite"
    )
    assert benchmark.name == "Test Benchmark"
    assert "gender" in benchmark.categories

# test_routers.py - API endpoint tests
def test_create_benchmark_endpoint(client, db_session):
    response = client.post("/api/v1/benchmarks/", json={
        "name": "Test Benchmark",
        "domain": "employment",
        "categories": ["gender"],
        "data_tier": "lite"
    })
    assert response.status_code == 201
    assert response.json()["name"] == "Test Benchmark"
```

#### SAGED Core Unit Tests

```python
# test_saged_data.py - Core functionality tests
def test_saged_data_initialization():
    saged = SAGEDData(
        domain="employment",
        concept="gender",
        data_tier="lite"
    )
    assert saged.domain == "employment"
    assert saged.concept == "gender"
    assert saged.data_tier == "lite"

# test_extractors.py - Feature extractor tests
def test_sentiment_extractor():
    extractor = SentimentExtractor()
    scores = extractor.extract(["I love this", "I hate this"])
    assert len(scores) == 2
    assert scores[0] > scores[1]  # First should be more positive
```

### 2. Integration Tests

**Purpose**: Test component interactions and workflows

```bash
# Run integration tests
python -m pytest tests/integration/ -v

# Run with database setup
python -m pytest tests/integration/ --setup-db
```

#### API Integration Tests

```python
# test_api_integration.py - Full API workflow tests
def test_benchmark_experiment_workflow(client, db_session):
    # Create benchmark
    benchmark_response = client.post("/api/v1/benchmarks/", json={
        "name": "Integration Test Benchmark",
        "domain": "employment",
        "categories": ["gender"],
        "data_tier": "lite"
    })
    benchmark_id = benchmark_response.json()["id"]

    # Create experiment
    experiment_response = client.post("/api/v1/experiments/", json={
        "name": "Integration Test Experiment",
        "benchmark_id": benchmark_id,
        "generation_config": {"model": "test"},
        "extraction_config": {"features": ["sentiment"]},
        "analysis_config": {"metrics": ["disparity"]}
    })
    experiment_id = experiment_response.json()["id"]

    # Run experiment
    run_response = client.post(f"/api/v1/experiments/{experiment_id}/run")
    assert run_response.status_code == 200

    # Check results
    results_response = client.get(f"/api/v1/analysis/experiments/{experiment_id}/results")
    assert results_response.status_code == 200
```

#### Database Integration Tests

```python
# test_database_integration.py - Database operations
def test_cascade_delete_experiment_with_results(db_session):
    # Create benchmark, experiment, and results
    benchmark = create_test_benchmark(db_session)
    experiment = create_test_experiment(db_session, benchmark.id)
    results = create_test_analysis_results(db_session, experiment.id)

    # Delete experiment should cascade to results
    db_session.delete(experiment)
    db_session.commit()

    # Verify results are deleted
    remaining_results = db_session.query(AnalysisResult).filter_by(
        experiment_id=experiment.id
    ).all()
    assert len(remaining_results) == 0
```

### 3. End-to-End Tests

**Purpose**: Test complete user workflows

```python
# test_full_pipeline.py - Complete bias detection workflow
def test_complete_bias_detection_pipeline():
    # Setup
    saged = SAGEDData(
        domain="employment",
        concept="gender",
        data_tier="lite"
    )

    # Run full pipeline
    results = saged.run_full_pipeline()

    # Validate results structure
    assert "bias_metrics" in results
    assert "group_analysis" in results
    assert "statistical_tests" in results

    # Validate bias detection
    bias_score = results["bias_metrics"]["overall_bias_score"]
    assert 0 <= bias_score <= 1

    # Validate statistical significance
    p_value = results["statistical_tests"]["t_test"]["p_value"]
    assert 0 <= p_value <= 1
```

### 4. Frontend Tests

**Purpose**: Test React components and user interactions

```bash
# Run frontend tests
cd app/frontend
npm test

# Run with coverage
npm run test:coverage

# Run in watch mode
npm run test:watch
```

```typescript
// Benchmarks.test.tsx - Component testing
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import Benchmarks from "../pages/Benchmarks";

test("creates new benchmark", async () => {
  const queryClient = new QueryClient();

  render(
    <QueryClientProvider client={queryClient}>
      <Benchmarks />
    </QueryClientProvider>
  );

  // Open create dialog
  fireEvent.click(screen.getByText("Create Benchmark"));

  // Fill form
  fireEvent.change(screen.getByLabelText("Name"), {
    target: { value: "Test Benchmark" },
  });

  fireEvent.change(screen.getByLabelText("Domain"), {
    target: { value: "employment" },
  });

  // Submit form
  fireEvent.click(screen.getByText("Create"));

  // Verify success
  await waitFor(() => {
    expect(
      screen.getByText("Benchmark created successfully")
    ).toBeInTheDocument();
  });
});
```

## 🔧 Test Configuration

### Pytest Configuration (`pytest.ini`)

```ini
[tool:pytest]
minversion = 6.0
addopts =
    -ra
    --strict-markers
    --disable-warnings
    --tb=short
    --cov=app
    --cov=saged
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=80

testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
    database: Tests requiring database
    external: Tests requiring external services
```

### Test Fixtures (`conftest.py`)

```python
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.backend.main import app
from app.backend.database import get_db, Base
from saged import SAGEDData

# Database fixtures
@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine."""
    engine = create_engine("sqlite:///./test.db")
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db_session(test_engine):
    """Create database session for tests."""
    TestingSessionLocal = sessionmaker(bind=test_engine)
    session = TestingSessionLocal()
    yield session
    session.close()

@pytest.fixture
def client(db_session):
    """Create test client with database override."""
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()

# SAGED fixtures
@pytest.fixture
def sample_saged():
    """Create sample SAGED instance for testing."""
    return SAGEDData(
        domain="employment",
        concept="gender",
        data_tier="lite"
    )

@pytest.fixture
def mock_llm_responses():
    """Mock LLM responses for testing."""
    return {
        "positive": "This is a great professional with excellent skills.",
        "negative": "This person might not be suitable for the role.",
        "neutral": "This is a standard professional profile."
    }

# Data fixtures
@pytest.fixture
def sample_benchmark_data():
    """Sample benchmark data for testing."""
    return {
        "name": "Test Gender Bias Benchmark",
        "description": "Testing gender bias in employment",
        "domain": "employment",
        "categories": ["gender"],
        "data_tier": "lite",
        "config": {
            "model": "test-model",
            "features": ["sentiment", "toxicity"]
        }
    }
```

## 📊 Test Data Management

### Test Fixtures and Mock Data

```python
# fixtures/sample_benchmarks.json
{
  "lite_benchmark": {
    "name": "Lite Gender Bias Test",
    "domain": "employment",
    "categories": ["gender"],
    "data_tier": "lite",
    "expected_samples": 50
  },
  "comprehensive_benchmark": {
    "name": "Comprehensive Bias Test",
    "domain": "healthcare",
    "categories": ["gender", "race", "age"],
    "data_tier": "scraped_sentences",
    "expected_samples": 1000
  }
}

# Using fixtures in tests
def test_benchmark_creation_with_fixture():
    with open("tests/fixtures/sample_benchmarks.json") as f:
        fixtures = json.load(f)

    benchmark_data = fixtures["lite_benchmark"]
    benchmark = create_benchmark(benchmark_data)

    assert benchmark.name == benchmark_data["name"]
    assert benchmark.data_tier == benchmark_data["data_tier"]
```

### Mock External Services

```python
# Mock LLM API responses
@pytest.fixture
def mock_openai_client(monkeypatch):
    """Mock OpenAI API client."""
    def mock_create(*args, **kwargs):
        return {
            "choices": [{
                "message": {
                    "content": "This is a mocked response."
                }
            }]
        }

    monkeypatch.setattr("openai.ChatCompletion.create", mock_create)

# Mock database operations
@pytest.fixture
def mock_database_error(monkeypatch):
    """Mock database connection error."""
    def mock_connect(*args, **kwargs):
        raise ConnectionError("Database unavailable")

    monkeypatch.setattr("sqlalchemy.create_engine", mock_connect)
```

## 🚀 Performance Testing

### Load Testing

```python
# test_performance.py - Performance and load tests
import time
import pytest
from concurrent.futures import ThreadPoolExecutor

@pytest.mark.slow
def test_concurrent_benchmark_creation(client):
    """Test concurrent benchmark creation."""
    def create_benchmark(index):
        return client.post("/api/v1/benchmarks/", json={
            "name": f"Concurrent Test {index}",
            "domain": "employment",
            "categories": ["gender"],
            "data_tier": "lite"
        })

    # Test 10 concurrent requests
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(create_benchmark, i) for i in range(10)]
        responses = [future.result() for future in futures]

    # All should succeed
    assert all(r.status_code == 201 for r in responses)

@pytest.mark.slow
def test_large_dataset_processing():
    """Test processing large datasets."""
    saged = SAGEDData(
        domain="employment",
        concept="gender",
        data_tier="questions"  # Large dataset
    )

    start_time = time.time()
    results = saged.run_full_pipeline()
    execution_time = time.time() - start_time

    # Should complete within reasonable time (5 minutes)
    assert execution_time < 300
    assert results["data_summary"]["total_prompts"] > 1000
```

### Memory Testing

```python
import psutil
import os

def test_memory_usage_during_processing():
    """Test memory usage doesn't exceed limits."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    saged = SAGEDData(
        domain="employment",
        concept="gender",
        data_tier="scraped_sentences"
    )

    results = saged.run_full_pipeline()

    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory

    # Memory increase should be reasonable (< 500MB)
    assert memory_increase < 500
```

## 🔍 Test Debugging

### Running Specific Tests

```bash
# Run single test
python -m pytest tests/unit/backend/test_models.py::test_benchmark_creation -v

# Run tests matching pattern
python -m pytest tests/ -k "test_benchmark" -v

# Run tests with specific markers
python -m pytest tests/ -m "unit" -v
python -m pytest tests/ -m "integration and not slow" -v

# Run failed tests from last run
python -m pytest --lf -v
```

### Debug Mode

```bash
# Run with debugging
python -m pytest tests/unit/test_specific.py --pdb

# Capture output
python -m pytest tests/ -s --capture=no

# Verbose output with variables
python -m pytest tests/ -vv --tb=long
```

### Test Coverage

```bash
# Generate coverage report
python -m pytest tests/ --cov=app --cov=saged --cov-report=html

# View coverage report
open htmlcov/index.html

# Coverage with missing lines
python -m pytest tests/ --cov=app --cov-report=term-missing

# Fail if coverage below threshold
python -m pytest tests/ --cov=app --cov-fail-under=85
```

## 🛠️ Writing New Tests

### Test Naming Conventions

```python
# Good test names
def test_benchmark_creation_with_valid_data():
def test_experiment_fails_with_invalid_benchmark_id():
def test_saged_data_initialization_with_custom_config():

# Bad test names
def test_benchmark():
def test_error():
def test_stuff():
```

### Test Structure (AAA Pattern)

```python
def test_benchmark_creation():
    # Arrange - Set up test data
    benchmark_data = {
        "name": "Test Benchmark",
        "domain": "employment",
        "categories": ["gender"],
        "data_tier": "lite"
    }

    # Act - Execute the functionality
    response = client.post("/api/v1/benchmarks/", json=benchmark_data)

    # Assert - Verify the results
    assert response.status_code == 201
    assert response.json()["name"] == benchmark_data["name"]
```

### Parameterized Tests

```python
@pytest.mark.parametrize("data_tier,expected_samples", [
    ("lite", 50),
    ("keywords", 200),
    ("scraped_sentences", 1000),
])
def test_saged_data_tiers(data_tier, expected_samples):
    saged = SAGEDData(
        domain="employment",
        concept="gender",
        data_tier=data_tier
    )

    saged.create_data()
    sample_count = len(saged.get_samples())

    assert sample_count >= expected_samples * 0.8  # Allow 20% variance
```

## 🚦 Continuous Integration

### GitHub Actions Configuration

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v3
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r tests/requirements-test.txt

      - name: Run unit tests
        run: python -m pytest tests/unit/ -v --cov=app --cov=saged

      - name: Run integration tests
        run: python -m pytest tests/integration/ -v

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: tests
        name: Run tests
        entry: python -m pytest tests/unit/ -x
        language: system
        always_run: true
        pass_filenames: false
```

## 📈 Test Metrics

### Coverage Targets

| Component               | Target Coverage | Current Coverage |
| ----------------------- | --------------- | ---------------- |
| **Backend API**         | 90%             | 87%              |
| **SAGED Core**          | 85%             | 82%              |
| **Database Models**     | 95%             | 93%              |
| **Frontend Components** | 80%             | 78%              |
| **Overall**             | 85%             | 83%              |

### Test Performance

| Test Category         | Target Time  | Current Time |
| --------------------- | ------------ | ------------ |
| **Unit Tests**        | < 30 seconds | 28 seconds   |
| **Integration Tests** | < 2 minutes  | 1.8 minutes  |
| **Full Suite**        | < 5 minutes  | 4.2 minutes  |

---

For more information, see the [main README](../README.md) or the [development setup guide](../setup_dev.sh).

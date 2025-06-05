# SAGED-Bias Testing

This directory contains the comprehensive test suite for the SAGED-Bias project, organized into unit and integration tests.

## Test Structure

### Unit Tests (`tests/unit/`)

- `test_saged_data.py` - Tests for the core SAGEDData class
- `test_scrape.py` - Tests for scraping functionality
- `test_utility.py` - Tests for utility functions
- `test_assembler.py` - Tests for PromptAssembler class
- `test_extractor.py` - Tests for FeatureExtractor class
- `test_diagnoser.py` - Tests for DisparityDiagnoser class
- `test_generator.py` - Tests for ResponseGenerator class

### Integration Tests (`tests/integration/`)

- `test_pipeline_integration.py` - Full pipeline integration tests
- `test_mpf_integration.py` - MPF extension integration tests

### Shared Fixtures (`conftest.py`)

- Common test fixtures and configuration
- Mock objects and sample data
- Test utilities and helpers

## Running Tests

### Using uv (Recommended)

```bash
# Run all tests
uv run pytest tests/

# Run only unit tests
uv run pytest tests/unit/

# Run only integration tests
uv run pytest tests/integration/

# Run tests with verbose output
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ --cov=saged --cov-report=term-missing

# Run tests with HTML coverage report
uv run pytest tests/ --cov=saged --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_extractor.py

# Run specific test class
uv run pytest tests/unit/test_extractor.py::TestFeatureExtractor

# Run specific test method
uv run pytest tests/unit/test_extractor.py::TestFeatureExtractor::test_extract_sentiment
```

### Using the Test Runner Script

```bash
# Basic test run
python run_tests.py

# Verbose output with coverage
python run_tests.py --verbose --coverage

# Run specific test file
python run_tests.py --specific test_saged_data.py

# Run specific test function
python run_tests.py --specific test_saged_data.py::test_init
```

## Coverage

Comprehensive test coverage across all major components:

### Core Components

- ✅ Core data structures and validation (SAGEDData)
- ✅ Utility functions
- ✅ Scraping functionality
- ✅ Prompt assembly and context building
- ✅ Feature extraction (sentiment, toxicity, stereotypes)
- ✅ Bias diagnosis and statistical analysis
- ✅ Response generation and LLM integration

### Pipeline Components

- ✅ Full pipeline orchestration
- ✅ Component interaction and data flow
- ✅ Error handling and recovery
- ✅ Configuration validation
- ✅ Performance optimization

### MPF Extension

- ✅ Multi-perspective fusion pipeline
- ✅ Bias mitigation strategies
- ✅ LLM factory and model management
- ✅ Integration with core SAGED components
- ✅ Reporting and visualization

## Adding New Tests

When adding new tests:

1. Follow the existing naming convention (`test_*.py`)
2. Use pytest fixtures for common test data
3. Mock external dependencies (APIs, file systems, etc.)
4. Test both success and failure cases
5. Update this README if adding new test categories

## Dependencies

Test dependencies are managed through uv and include:

- pytest >= 8.4.0
- pytest-cov >= 6.1.1
- All production dependencies

## Environment Setup

```bash
# Create virtual environment
uv venv --python 3.10

# Install dependencies
uv sync

# Add test dependencies (if not already included)
uv add pytest pytest-cov
```

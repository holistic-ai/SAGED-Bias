# SAGED: A Holistic Bias-Benchmarking Pipeline for Language Models with Customisable Fairness Calibration

[![ArXiv](https://img.shields.io/badge/ArXiv-2409.11149-red)](https://arxiv.org/abs/2409.11149)
![License](https://img.shields.io/badge/License-MIT-blue)

**Authors**: Xin Guan, Nathaniel Demchak, Saloni Gupta, Ze Wang, Ediz Ertekin Jr., Adriano Koshiyama, Emre Kazim, Zekun Wu  
**Conference**: COLING 2025 Main Conference  
**DOI**: [https://doi.org/10.48550/arXiv.2409.11149](https://doi.org/10.48550/arXiv.2409.11149)

---

## Overview

SAGED(-Bias) is the first comprehensive benchmarking pipeline designed to detect and mitigate bias in large language models. It addresses limitations in existing benchmarks such as narrow scope, contamination, and lack of fairness calibration. The SAGED pipeline includes the following five core stages:

![System Diagram](diagrams/pipeline.png)

This diagram illustrates the core stages of the SAGED pipeline:

1. **Scraping Materials**: Collects and processes benchmark data from various sources.
2. **Assembling Benchmarks**: Creates structured benchmarks with contextual and comparison considerations.
3. **Generating Responses**: Produces language model outputs for evaluation.
4. **Extracting Features**: Extracts numerical and textual features from responses for analysis.
5. **Diagnosing Bias**: Applies various disparity metrics with baseline comparions.

---

## Installation

### Install the library from PyPI:

```bash
pip install sagedbias
```

### Development Setup

For development and testing, we recommend using `uv` for fast dependency management:

```bash
# Install uv
pip install uv

# Clone the repository
git clone https://github.com/holistic-ai/SAGED-Bias.git
cd SAGED-Bias

# Create virtual environment and install dependencies
uv venv --python 3.10
uv sync

# Run tests
uv run pytest tests/
```

## Quick Start

### Basic Usage

```python
from saged import Pipeline
from saged import Scraper, KeywordFinder, SourceFinder
from saged import PromptAssembler
from saged import FeatureExtractor
from saged import DisparityDiagnoser

# Build a bias benchmark
config = {
    'categories': ['nationality'],
    'branching': True,
    'shared_config': {
        'keyword_finder': {'require': True},
        'source_finder': {'require': True},
        'scraper': {'require': True},
        'prompt_assembler': {'require': True}
    }
}

benchmark = Pipeline.build_benchmark('demographics', config)
```

### Research Extensions (MPF)

For advanced research using Multi-Perspective Fusion (MPF):

```python
from saged.mpf import mpf_pipeline, Mitigator, LLMFactory

# MPF bias mitigation
mitigator = Mitigator()
results = mpf_pipeline(benchmark_data, mitigator)
```

## Repository Structure

```
SAGED-Bias/
├── saged/                    # Core SAGED pipeline
│   ├── mpf/                  # MPF extension (Multi-Perspective Fusion)
│   ├── _pipeline.py          # Main pipeline
│   ├── _extractor.py         # Feature extraction
│   └── ...                   # Other core modules
├── tests/                    # Test suite (21 tests)
├── tutorials/                # Jupyter notebook tutorials
├── examples/                 # Usage examples
├── scripts/                  # Utility scripts
└── MPF_icml/                 # ICML 2025 paper materials
```

## Testing

Run the test suite:

```bash
# Using uv (recommended)
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=saged

# Using the test runner script
python run_tests.py --verbose --coverage
```

## Maintenance

Clean build artifacts and temporary files:

```bash
# Preview cleanup
python scripts/cleanup.py --dry-run

# Clean cache files and build artifacts
python scripts/cleanup.py

# Clean everything including coverage reports
python scripts/cleanup.py --all
```

## Citation

If you use SAGED in your work, please cite the following paper:

```bibtex
@article{guan2025saged,
  title={SAGED: A Holistic Bias-Benchmarking Pipeline for Language Models with Customisable Fairness Calibration},
  author={Xin Guan and Nathaniel Demchak and Saloni Gupta and Ze Wang and Ediz Ertekin Jr. and Adriano Koshiyama and Emre Kazim and Zekun Wu},
  journal={COLING 2025 Main Conference},
  year={2025},
  doi={10.48550/arXiv.2409.11149}
}
```

## License

SAGED-bias is released under the MIT License.

# SAGED: A Holistic Bias-Benchmarking Pipeline for Language Models with Customisable Fairness Calibration


[![ArXiv](https://img.shields.io/badge/ArXiv-2409.11149-red)](https://arxiv.org/abs/2409.11149) 
![License](https://img.shields.io/badge/License-MIT-blue)

**Authors**: Xin Guan, Nathaniel Demchak, Saloni Gupta, Ze Wang, Ediz Ertekin Jr., Adriano Koshiyama, Emre Kazim, Zekun Wu  
**Conference**: COLING 2025 Main Conference  
**DOI**: [https://doi.org/10.48550/arXiv.2409.11149](https://doi.org/10.48550/arXiv.2409.11149)

---

## Overview

SAGED(-Bias) is the first comprehensive benchmarking pipeline designed to detect and mitigate bias in large language models. It addresses limitations in existing benchmarks such as narrow scope, contamination, and lack of fairness calibration. The SAGED pipeline includes the following five core stages:

![System Diagram](system_diagram.png)

This diagram illustrates the core stages of the SAGED pipeline:

1. **Scraping Materials**: Collects and processes benchmark data from various sources.
2. **Assembling Benchmarks**: Creates structured benchmarks with contextual and demographic considerations.
3. **Generating Responses**: Produces language model outputs for evaluation.
4. **Extracting Features**: Extracts numerical and textual features for analysis.
5. **Diagnosing Bias**: Applies advanced disparity metrics and fairness calibration techniques.

SAGED evaluates max disparity (e.g., impact ratio) and bias concentration (e.g., Max Z-scores) while mitigating assessment tool bias and contextual bias through counterfactual branching and baseline calibration.

---

## Installation

### Install the latest version of SAGED-bias from PyPi using pip:

```bash
pip install sagedbias
```

### Or install the development version from GitHub:

```bash
# Clone the repository
git clone https://github.com/holistic-ai/SAGED-Bias.git
cd SAGED-Bias

# Install Hatch (if not already installed)
pip install hatch

# Create and activate a virtual environment
hatch env create
hatch shell
```
 
### Install the dependencies
```bash
hatch run install
```

### Running Tests
```bash
hatch run pytest tests --cache-clear --cov=saged --cov-report=term
```

## Key Features

### 1. Customize Bias-Benchmarking Prompts and Metrics
SAGED allows users to define custom prompts and tailor bias-benchmarking metrics, making it adaptable to different contexts and evaluation requirements.

### 2. Benchmark Building: Scrape and Assemble
- **Scraping (`_scrape.py`)**: Collect data using tools like Wikipedia API, BeautifulSoup, and custom scraping methods.
- **Assembling (`_assembler.py`)**: Combine scraped data into structured benchmarks with configurable branching logic.

### 3. Benchmark Running: Generate and Extract
- **Generate (`_generator.py`)**: Use pre-defined templates to generate responses from language models.
- **Extract (`_extractor.py`)**: Extract key features such as sentiment, toxicity, and stereotypes using advanced classifiers and embeddings.

### 4. Diagnosis: Group, Summarize, and Compare
- **Diagnose (`_diagnoser.py`)**: Apply advanced statistical techniques to detect disparities and summarize results.
- **Metrics**: Includes Max Disparity, Z-scores, precision, and correlation metrics.

### 5. Pipeline - Build and Run Benchmark
- **Pipeline (`_pipeline.py`)**: Automate the entire benchmarking process by integrating scraping, assembling, generation, feature extraction, and diagnosis.

## Usage Guide

### Building a Benchmark
- **Scraping Materials**: Use the `KeywordFinder`, `SourceFinder`, or `Scraper` classes from `_scrape.py` to collect benchmark data.
- **Assembling Prompts**: Use the `PromptAssembler` class in `_assembler.py` to split sentences and create custom prompts.

### Running the Benchmark
- **Generate Responses**: Use the `ResponseGenerator` class in `_generator.py` to generate outputs from language models.
- **Extract Features**: Apply the `FeatureExtractor` class in `_extractor.py` for sentiment, toxicity, and stereotype analysis.

### Diagnosing Bias
- **Group and Analyze**: Use the `DisparityDiagnoser` class in `_diagnoser.py` to calculate group statistics and compare disparities.
- **Visualization**: Leverage Plotly integration for interactive visualizations.

### End-to-End Pipeline
- The `Pipeline` class in `_pipeline.py` integrates all stages into a seamless workflow.


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

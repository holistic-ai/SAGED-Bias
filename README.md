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

[//]: # (### Install the latest version of SAGED-bias from PyPi using pip:)

[//]: # ()
[//]: # (```bash)

[//]: # (pip install sagedbias)

[//]: # (```)

### Install the library from PyPI:

```bash
pip install sagedbias
```
 
### import the library 
```python
from saged import Pipeline
from saged import Scraper, KeywordFinder, SourceFinder
from saged import PromptAssembler
from saged import FeatureExtractor
from saged import DisparityDiagnoser
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

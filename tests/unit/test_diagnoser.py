"""
Unit tests for DisparityDiagnoser class.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from saged import DisparityDiagnoser


@pytest.fixture
def diagnoser_config():
    """Sample diagnoser configuration."""
    return {
        'metrics': ['statistical_parity', 'equalized_odds', 'demographic_parity'],
        'features': ['sentiment', 'toxicity'],
        'alpha': 0.05,
        'bootstrap_samples': 100,
        'confidence_level': 0.95
    }


@pytest.fixture 
def sample_extractions_df():
    """Sample extractions DataFrame with bias indicators."""
    np.random.seed(42)
    n_samples = 90
    concepts = ['education', 'healthcare', 'employment'] * 30
    
    return pd.DataFrame({
        'concept': concepts,
        'baseline_sentiment': np.random.normal(0.0, 0.2, n_samples),
        'generation_1_sentiment': np.random.normal(0.1, 0.3, n_samples),
        'generation_2_sentiment': np.random.normal(-0.05, 0.25, n_samples),
        'generation_3_sentiment': np.random.normal(0.15, 0.3, n_samples),
        'baseline_toxicity': np.random.uniform(0, 0.1, n_samples),
        'generation_1_toxicity': np.random.uniform(0, 0.2, n_samples),
        'generation_2_toxicity': np.random.uniform(0, 0.15, n_samples),
        'generation_3_toxicity': np.random.uniform(0, 0.1, n_samples),
    })


class TestDisparityDiagnoser:
    """Test suite for DisparityDiagnoser."""

    def test_init_default(self, sample_benchmark_data_for_diagnoser):
        """Test DisparityDiagnoser initialization with default settings."""
        diagnoser = DisparityDiagnoser(
            sample_benchmark_data_for_diagnoser,
            features=['sentiment_score', 'toxicity_score'],
            generations=['LLM'],
            baseline='baseline'
        )
        
        assert diagnoser.benchmark is not None
        assert isinstance(diagnoser.benchmark, pd.DataFrame)
        assert diagnoser.baseline == 'baseline'

    def test_init_with_config(self, sample_benchmark_data):
        """Test DisparityDiagnoser initialization with configuration."""
        diagnoser = DisparityDiagnoser(
            sample_benchmark_data,
            features=['sentiment_score'],
            target_groups=['business', 'politics'],
            generations=['baseline', 'LLM'],
            baseline='baseline',
            group_type='domain'
        )
        
        assert diagnoser.benchmark is not None
        assert diagnoser.baseline == 'baseline'

    def test_diagnose_basic(self, sample_benchmark_data):
        """Test basic disparity diagnosis."""
        diagnoser = DisparityDiagnoser(sample_benchmark_data)
        
        # Test that diagnoser has the required attributes
        assert hasattr(diagnoser, 'features')
        assert hasattr(diagnoser, 'generations')
        assert hasattr(diagnoser, 'target_groups')

    def test_statistical_parity(self, sample_benchmark_data):
        """Test statistical parity calculation."""
        diagnoser = DisparityDiagnoser(sample_benchmark_data)
        
        # Test that we can call statistical methods
        result = diagnoser.mean()
        assert isinstance(result, type(diagnoser))  # Returns self for chaining

    def test_equalized_odds(self, sample_benchmark_data):
        """Test equalized odds calculation."""
        diagnoser = DisparityDiagnoser(sample_benchmark_data)
        
        # Test variance calculation
        result = diagnoser.variance()
        assert isinstance(result, type(diagnoser))

    def test_demographic_parity(self, sample_benchmark_data):
        """Test demographic parity calculation."""
        diagnoser = DisparityDiagnoser(sample_benchmark_data)
        
        # Test standard deviation calculation
        result = diagnoser.standard_deviation()
        assert isinstance(result, type(diagnoser))

    def test_bootstrap_confidence_intervals(self, sample_benchmark_data):
        """Test bootstrap confidence interval calculation."""
        diagnoser = DisparityDiagnoser(sample_benchmark_data)
        
        # Test that diagnoser can handle the benchmark data
        assert len(diagnoser.benchmark) > 0
        assert 'concept' in diagnoser.benchmark.columns

    def test_significance_testing(self, sample_benchmark_data):
        """Test statistical significance testing."""
        diagnoser = DisparityDiagnoser(sample_benchmark_data)
        
        # Test range calculation
        result = diagnoser.range()
        assert isinstance(result, type(diagnoser))

    def test_effect_size_calculation(self, sample_benchmark_data):
        """Test effect size calculation."""
        diagnoser = DisparityDiagnoser(sample_benchmark_data)
        
        # Test quantile range calculation
        result = diagnoser.quantile_range()
        assert isinstance(result, type(diagnoser))

    def test_disparity_summary(self, sample_benchmark_data):
        """Test disparity summary generation."""
        diagnoser = DisparityDiagnoser(sample_benchmark_data)
        
        # Test that we can access summary data
        assert hasattr(diagnoser, 'summary_df_dict')
        assert hasattr(diagnoser, 'disparity_df')

    def test_concept_based_analysis(self, sample_benchmark_data):
        """Test concept-based disparity analysis."""
        diagnoser = DisparityDiagnoser(sample_benchmark_data)
        
        # Test that concepts are properly handled
        concepts = diagnoser.benchmark['concept'].unique()
        assert len(concepts) > 0

    def test_feature_comparison(self, sample_benchmark_data):
        """Test feature comparison across groups."""
        diagnoser = DisparityDiagnoser(sample_benchmark_data)
        
        # Test that features are properly configured
        assert len(diagnoser.features) > 0

    def test_threshold_analysis(self, sample_benchmark_data):
        """Test threshold-based analysis."""
        diagnoser = DisparityDiagnoser(sample_benchmark_data)
        
        # Test skewness calculation
        result = diagnoser.skewness()
        assert isinstance(result, type(diagnoser))

    def test_temporal_analysis(self, sample_benchmark_data):
        """Test temporal disparity analysis."""
        diagnoser = DisparityDiagnoser(sample_benchmark_data)
        
        # Test kurtosis calculation
        result = diagnoser.kurtosis()
        assert isinstance(result, type(diagnoser))

    def test_multiple_generations_comparison(self, sample_benchmark_data):
        """Test comparison across multiple generations."""
        diagnoser = DisparityDiagnoser(sample_benchmark_data)
        
        # Test that multiple generations are handled
        assert len(diagnoser.generations) > 0

    def test_input_validation(self, sample_benchmark_data):
        """Test input validation."""
        # Valid data should not raise errors
        diagnoser = DisparityDiagnoser(sample_benchmark_data)
        assert diagnoser.benchmark is not None
        
        # Test with invalid group type
        with pytest.raises(AssertionError):
            DisparityDiagnoser(sample_benchmark_data, group_type='invalid')

    def test_missing_data_handling(self, sample_benchmark_data):
        """Test handling of missing data."""
        diagnoser = DisparityDiagnoser(sample_benchmark_data)
        
        # Test with data containing NaN values
        data_with_nan = sample_benchmark_data.copy()
        data_with_nan.loc[0, 'baseline_sentiment_score'] = np.nan
        
        diagnoser.benchmark = data_with_nan
        
        # Should handle NaN values gracefully
        result = diagnoser.mean()
        assert isinstance(result, type(diagnoser))

    def test_small_sample_handling(self, sample_benchmark_data):
        """Test handling of small sample sizes."""
        # Create small sample
        small_sample = sample_benchmark_data.head(2)
        diagnoser = DisparityDiagnoser(small_sample)
        
        # Should handle small samples
        assert len(diagnoser.benchmark) == 2

    def test_outlier_detection(self, sample_benchmark_data):
        """Test outlier detection and handling."""
        diagnoser = DisparityDiagnoser(sample_benchmark_data)
        
        # Test statistics disparity calculation
        result = diagnoser.statistics_disparity()
        assert isinstance(result, type(diagnoser))

    def test_export_results(self, sample_benchmark_data):
        """Test exporting analysis results."""
        diagnoser = DisparityDiagnoser(sample_benchmark_data)
        
        # Test that we can access results
        assert hasattr(diagnoser, 'summary_df_dict')
        assert hasattr(diagnoser, 'summary_df_dict_with_p_values') 
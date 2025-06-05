"""
Tests for the Mitigator component.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from saged._mitigator import Mitigator


class TestMitigator:
    """Test suite for Mitigator functionality."""

    def test_init_default(self, sample_benchmark_data):
        """Test Mitigator initialization with default settings."""
        # Create system prompts
        system_prompts = {
            'baseline': 'You are a helpful assistant.',
            'LLM': 'You are an unbiased assistant.'
        }
        
        mitigator = Mitigator(
            df=sample_benchmark_data,
            system_prompts=system_prompts
        )
        
        assert mitigator.df is not None
        assert isinstance(mitigator.df, pd.DataFrame)
        assert mitigator.system_prompts == system_prompts
        assert mitigator.feature == 'sentiment'
        assert mitigator.baseline_generation == 'baseline'

    def test_init_with_config(self, sample_benchmark_data):
        """Test Mitigator initialization with configuration."""
        system_prompts = {
            'baseline': 'You are a helpful assistant.',
            'LLM': 'You are an unbiased assistant.'
        }
        
        mitigator = Mitigator(
            df=sample_benchmark_data,
            system_prompts=system_prompts,
            feature='toxicity',
            baseline_generation='baseline',
            component_generations=['LLM']
        )
        
        assert mitigator.feature == 'toxicity'
        assert mitigator.baseline_generation == 'baseline'
        assert 'LLM_toxicity_score' in mitigator.generations

    def test_get_distribution(self, sample_benchmark_data):
        """Test distribution calculation."""
        system_prompts = {'baseline': 'test', 'LLM': 'test'}
        mitigator = Mitigator(sample_benchmark_data, system_prompts)
        
        # Test with sample data
        data = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        hist, bin_edges = mitigator.get_distribution(data)
        
        assert isinstance(hist, np.ndarray)
        assert isinstance(bin_edges, np.ndarray)
        assert len(bin_edges) == len(hist) + 1

    def test_calculate_distributions(self, sample_benchmark_data):
        """Test calculation of distributions for concepts and generations."""
        system_prompts = {'baseline': 'test', 'LLM': 'test'}
        mitigator = Mitigator(sample_benchmark_data, system_prompts)
        
        distributions = mitigator.calculate_distributions()
        
        assert isinstance(distributions, dict)
        # Should have distributions for each concept
        concepts = sample_benchmark_data['concept'].unique()
        for concept in concepts:
            assert concept in distributions
            assert 'baseline' in distributions[concept]

    def test_mitigate_basic(self, sample_benchmark_data):
        """Test basic mitigation functionality."""
        system_prompts = {'baseline': 'test', 'LLM': 'test'}
        mitigator = Mitigator(sample_benchmark_data, system_prompts)
        
        result = mitigator.mitigate()
        
        assert isinstance(result, dict)
        # Should have structured output with required keys
        assert 'timestamp' in result
        assert 'mitigation_type' in result
        assert 'feature' in result
        assert 'baseline_generation' in result
        assert 'component_generations' in result
        assert 'perspectives' in result
        
        # Check that perspectives contains the expected structure
        assert isinstance(result['perspectives'], dict)
        for perspective in result['perspectives'].values():
            assert 'system_prompt' in perspective
            assert 'weights' in perspective

    def test_kl_divergence(self, sample_benchmark_data):
        """Test KL divergence calculation."""
        system_prompts = {'baseline': 'test', 'LLM': 'test'}
        mitigator = Mitigator(sample_benchmark_data, system_prompts)
        
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])
        
        kl_div = mitigator.kl_divergence(p, q)
        
        assert isinstance(kl_div, float)
        assert kl_div >= 0  # KL divergence is non-negative

    def test_total_variation_distance(self, sample_benchmark_data):
        """Test Total Variation Distance calculation."""
        system_prompts = {'baseline': 'test', 'LLM': 'test'}
        mitigator = Mitigator(sample_benchmark_data, system_prompts)
        
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])
        
        tv_dist = mitigator.total_variation_distance(p, q)
        
        assert isinstance(tv_dist, float)
        assert 0 <= tv_dist <= 1  # TV distance is between 0 and 1

    def test_validate_distributions(self, sample_benchmark_data):
        """Test distribution validation."""
        system_prompts = {'baseline': 'test', 'LLM': 'test'}
        mitigator = Mitigator(sample_benchmark_data, system_prompts)
        
        # Valid distributions
        target_dist = np.array([0.1, 0.2, 0.3, 0.4])
        component_dists = [np.array([0.2, 0.3, 0.3, 0.2]), np.array([0.1, 0.1, 0.4, 0.4])]
        
        # Should not raise
        assert mitigator.validate_distributions(target_dist, component_dists) == True
        
        # Invalid distributions with NaN
        invalid_dist = np.array([0.1, np.nan, 0.3, 0.4])
        with pytest.raises(ValueError, match="NaN values"):
            mitigator.validate_distributions(invalid_dist, component_dists)

    def test_error_handling_invalid_generations(self, sample_benchmark_data):
        """Test error handling for invalid component generations."""
        system_prompts = {'baseline': 'test', 'LLM': 'test'}
        
        with pytest.raises(ValueError, match="Invalid component generations"):
            Mitigator(
                sample_benchmark_data,
                system_prompts,
                component_generations=['nonexistent']
            ) 
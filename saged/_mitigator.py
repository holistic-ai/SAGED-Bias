import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance, entropy
from scipy.optimize import minimize
from typing import List, Dict, Tuple
import os
import json
from datetime import datetime

try:
    from ._plotter import Plotter
except ImportError:
    from ._plotter import Plotter

class Mitigator:
    # ===== Core Initialization and Setup =====
    def __init__(self, df: pd.DataFrame, system_prompts: Dict[str, str], output_dir: str = 'data/xntion/mitigator', 
                 feature: str = 'sentiment', baseline_generation: str = 'baseline', 
                 component_generations: List[str] = None):
        """
        Initialize the Mitigator with a DataFrame and system prompts
        
        Args:
            df (pd.DataFrame): DataFrame containing the sentiment scores
            system_prompts (Dict[str, str]): Dictionary mapping perspective names to their system prompts
            output_dir (str): Directory to save output files (default: 'data/xntion/mitigator')
            feature (str): The feature being analyzed (default: 'sentiment')
            baseline_generation (str): Name of the baseline generation to use as target (default: 'baseline')
            component_generations (List[str]): List of generation names to use as components. 
                                             If None, uses all generations except baseline.
        """
        self.df = df
        self.system_prompts = system_prompts
        self.feature = feature
        self.baseline_generation = baseline_generation
        
        # Get all possible generations
        all_generations = [col.replace(f'_{self.feature}_score', '') for col in self.df.columns 
                          if col.endswith(f'_{self.feature}_score') and not col.startswith(self.baseline_generation)]
        
        # Set component generations
        if component_generations is None:
            self.component_generations = all_generations
        else:
            # Validate that all specified components exist
            invalid_components = [gen for gen in component_generations if gen not in all_generations]
            if invalid_components:
                raise ValueError(f"Invalid component generations: {invalid_components}. "
                               f"Available generations are: {all_generations}")
            self.component_generations = component_generations
            
        # Convert component generation names to column names
        self.generations = [f"{gen}_{self.feature}_score" for gen in self.component_generations]
        self.concepts = self.df['concept'].unique()
        
        # Add timestamp to output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(output_dir, f'run_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        self.plotter = Plotter(self.output_dir)

    # ===== Distribution Calculation and Validation =====
    def get_distribution(self, data: pd.Series, bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate histogram distribution of sentiment scores"""
        hist, bin_edges = np.histogram(data.dropna(), bins=bins, density=True)
        return hist, bin_edges
    
    def calculate_distributions(self) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """Calculate distributions for each concept and generation"""
        distributions = {}
        for concept in self.concepts:
            concept_data = self.df[self.df['concept'] == concept]
            distributions[concept] = {}
            
            # Calculate baseline distribution
            baseline_dist = self.get_distribution(concept_data[f'{self.baseline_generation}_{self.feature}_score'])
            distributions[concept][self.baseline_generation] = baseline_dist
            
            # Calculate distributions for each generation
            for gen in self.generations:
                gen_dist = self.get_distribution(concept_data[gen])
                distributions[concept][gen] = gen_dist
                
        return distributions

    def calculate_domain_distributions(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Calculate distributions at domain level directly from domain data"""
        domain_distributions = {}
        for domain in self.df['domain'].unique():
            domain_data = self.df[self.df['domain'] == domain]
            domain_distributions[domain] = {}
            
            # Calculate baseline distribution for the domain
            baseline_dist = self.get_distribution(domain_data[f'{self.baseline_generation}_{self.feature}_score'])
            domain_distributions[domain][self.baseline_generation] = baseline_dist[0]  # Only store histogram values
            
            # Calculate distributions for each generation
            for gen in self.generations:
                gen_dist = self.get_distribution(domain_data[gen])
                domain_distributions[domain][gen] = gen_dist[0]  # Only store histogram values
                
        return domain_distributions

    def validate_distributions(self, target_dist: np.ndarray, 
                             component_dists: List[np.ndarray],
                             is_calibration: bool = False) -> bool:
        """Validate input distributions before optimization"""
        # Check for NaN values
        if np.isnan(target_dist).any() or any(np.isnan(dist).any() for dist in component_dists):
            raise ValueError("Input distributions contain NaN values")
        
        # For calibration, we don't check for negative values since differences can be negative
        if not is_calibration:
            # Check for negative values
            if (target_dist < 0).any() or any((dist < 0).any() for dist in component_dists):
                raise ValueError("Input distributions contain negative values")
            
            # Check for zero distributions
            if np.all(target_dist == 0) or any(np.all(dist == 0) for dist in component_dists):
                raise ValueError("Input distributions contain all zeros")
        
        return True

    # ===== Distance/Divergence Metrics =====
    def kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate KL divergence between two distributions"""
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        return entropy(p, q)
    
    def total_variation_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Total Variation Distance between two distributions"""
        return 0.5 * np.sum(np.abs(p - q))

    # ===== Distribution Combination Methods =====
    def combine_distributions_bma(self, distributions: List[np.ndarray], 
                                weights: np.ndarray = None,
                                prior_mean: float = None,
                                prior_variance: float = None) -> np.ndarray:
        """Combine distributions using proper Bayesian Model Averaging (BMA)"""
        if weights is None:
            weights = np.ones(len(distributions)) / len(distributions)
            
        # If prior parameters not provided, use baseline statistics
        if prior_mean is None or prior_variance is None:
            baseline_dist = self.df[f'{self.baseline_generation}_{self.feature}_score']
            if prior_mean is None:
                prior_mean = baseline_dist.mean()
            if prior_variance is None:
                prior_variance = baseline_dist.var(ddof=0)
        
        # Calculate model-specific parameters
        model_params = []
        for dist in distributions:
            model_mean = np.mean(dist)
            model_variance = np.var(dist, ddof=0)
            model_params.append((model_mean, model_variance))
        
        # Calculate marginal likelihoods for each model
        marginal_likelihoods = []
        for (model_mean, model_variance) in model_params:
            likelihood = np.exp(-0.5 * ((model_mean - prior_mean)**2) / (model_variance + prior_variance))
            likelihood /= np.sqrt(2 * np.pi * (model_variance + prior_variance))
            marginal_likelihoods.append(likelihood)
        
        # Calculate posterior model probabilities
        marginal_likelihoods = np.array(marginal_likelihoods)
        posterior_probs = marginal_likelihoods * weights
        posterior_probs = posterior_probs / np.sum(posterior_probs)
        
        # Combine predictions using posterior probabilities
        weighted_sum = np.zeros_like(distributions[0])
        for prob, dist in zip(posterior_probs, distributions):
            weighted_sum += prob * dist
            
        return weighted_sum

    def apply_equivalent_weights(self, distributions: List[np.ndarray],
                               equivalent_weights: np.ndarray,
                               prior_mean: float) -> np.ndarray:
        """Apply equivalent weights to get combined distribution using simple weighted average"""
        # First element is prior weight
        prior_weight = equivalent_weights[0]
        dist_weights = equivalent_weights[1:]
        
        # Calculate weighted sum
        weighted_sum = np.zeros_like(distributions[0])
        weighted_sum += prior_mean * prior_weight
        for w, dist in zip(dist_weights, distributions):
            weighted_sum += w * dist
            
        return weighted_sum

    def get_equivalent_weights(self, distributions: List[np.ndarray],
                             bma_weights: np.ndarray,
                             prior_mean: float = None,
                             prior_variance: float = None) -> np.ndarray:
        """Convert BMA weights to equivalent simple weighted sum weights"""
        # If prior parameters not provided, use baseline statistics
        if prior_mean is None or prior_variance is None:
            baseline_dist = self.df[f'{self.baseline_generation}_{self.feature}_score']
            if prior_mean is None:
                prior_mean = baseline_mean = baseline_dist.mean()
            if prior_variance is None:
                prior_variance = baseline_dist.var(ddof=0)
        
        # Calculate model-specific parameters
        model_params = []
        for dist in distributions:
            model_mean = np.mean(dist)
            model_variance = np.var(dist, ddof=0)
            model_params.append((model_mean, model_variance))
        
        # Calculate marginal likelihoods for each model
        marginal_likelihoods = []
        for (model_mean, model_variance) in model_params:
            likelihood = np.exp(-0.5 * ((model_mean - prior_mean)**2) / (model_variance + prior_variance))
            likelihood /= np.sqrt(2 * np.pi * (model_variance + prior_variance))
            marginal_likelihoods.append(likelihood)
        
        # Calculate posterior model probabilities
        marginal_likelihoods = np.array(marginal_likelihoods)
        posterior_probs = marginal_likelihoods * bma_weights
        posterior_probs = posterior_probs / np.sum(posterior_probs)
        
        # Calculate equivalent weights for simple weighted sum
        means = np.array([mean for mean, _ in model_params])
        target_mean = np.sum(posterior_probs * means)
        
        # Solve for equivalent weights that give the same mean
        n_models = len(means)
        A = np.vstack([means, np.ones(n_models)])
        b = np.array([target_mean, 1.0])
        
        # Ensure dimensions match
        if A.shape[1] != n_models:
            A = A.T
            
        # Solve the system
        try:
            equivalent_weights = np.linalg.lstsq(A, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Fallback to simple normalization if linear system fails
            equivalent_weights = posterior_probs
        
        # Ensure weights sum to 1 and are non-negative
        equivalent_weights = np.maximum(equivalent_weights, 0)
        equivalent_weights = equivalent_weights / np.sum(equivalent_weights)
        
        # Round to 3 decimal places and renormalize
        equivalent_weights = np.array([float(f"{w:.3f}") for w in equivalent_weights])
        equivalent_weights = equivalent_weights / np.sum(equivalent_weights)
        
        return equivalent_weights

    # ===== Objective Functions =====
    def objective_function_wasserstein(self, weights: np.ndarray, 
                                     target_dist: np.ndarray, 
                                     component_dists: List[np.ndarray],
                                     alpha: float = 0,
                                     beta: float = 0,
                                     use_bma: bool = False) -> float:
        """Calculate Wasserstein distance with regularization"""
        weights = weights / np.sum(weights)
        
        if use_bma:
            # Get baseline statistics for prior
            baseline_mean = np.mean(target_dist)
            baseline_var = np.var(target_dist, ddof=0)
            
            # Get combined distribution using BMA
            weighted_sum = self.combine_distributions_bma(
                component_dists,
                weights,
                prior_mean=baseline_mean,
                prior_variance=baseline_var
            )
        else:
            weighted_sum = np.zeros_like(target_dist)
            for w, dist in zip(weights, component_dists):
                weighted_sum += w * dist
        
        wasserstein = wasserstein_distance(target_dist, weighted_sum)
        
        # Regularization terms
        uniform_weights = np.ones_like(weights) / len(weights)
        l2_regularization = alpha * np.sum((weights - uniform_weights) ** 2)
        
        # Concentration penalty
        n_nonzero = np.sum(weights > 1e-3)
        max_weight = np.max(weights)
        concentration_penalty = beta * (n_nonzero / len(weights) + (1 - max_weight))
        
        return wasserstein + l2_regularization + concentration_penalty
    
    def objective_function_kl(self, weights: np.ndarray, 
                            target_dist: np.ndarray, 
                            component_dists: List[np.ndarray],
                            alpha: float = 0,
                            beta: float = 0,
                            use_bma: bool = False) -> float:
        """Calculate KL divergence with regularization"""
        weights = weights / np.sum(weights)
        if use_bma:
            baseline_mean = np.mean(target_dist)
            baseline_var = np.var(target_dist, ddof=0)
            weighted_sum = self.combine_distributions_bma(
                component_dists,
                weights,
                prior_mean=baseline_mean,
                prior_variance=baseline_var
            )
        else:
            weighted_sum = np.zeros_like(target_dist)
            for w, dist in zip(weights, component_dists):
                weighted_sum += w * dist
        
        kl_div = self.kl_divergence(target_dist, weighted_sum)
        
        # Regularization terms
        uniform_weights = np.ones_like(weights) / len(weights)
        l2_regularization = alpha * np.sum((weights - uniform_weights) ** 2)
        
        # Concentration penalty
        n_nonzero = np.sum(weights > 1e-3)
        max_weight = np.max(weights)
        concentration_penalty = beta * (n_nonzero / len(weights) + (1 - max_weight))
        
        return kl_div + l2_regularization + concentration_penalty
    
    def objective_function_tv(self, weights: np.ndarray, 
                            target_dist: np.ndarray, 
                            component_dists: List[np.ndarray],
                            alpha: float = 0,
                            beta: float = 0,
                            use_bma: bool = False) -> float:
        """Calculate Total Variation Distance with regularization"""
        weights = weights / np.sum(weights)
        if use_bma:
            baseline_mean = np.mean(target_dist)
            baseline_var = np.var(target_dist, ddof=0)
            weighted_sum = self.combine_distributions_bma(
                component_dists,
                weights,
                prior_mean=baseline_mean,
                prior_variance=baseline_var
            )
        else:
            weighted_sum = np.zeros_like(target_dist)
            for w, dist in zip(weights, component_dists):
                weighted_sum += w * dist
        
        tv_dist = self.total_variation_distance(target_dist, weighted_sum)
        
        # Regularization terms
        uniform_weights = np.ones_like(weights) / len(weights)
        l2_regularization = alpha * np.sum((weights - uniform_weights) ** 2)
        
        # Concentration penalty
        n_nonzero = np.sum(weights > 1e-3)
        max_weight = np.max(weights)
        concentration_penalty = beta * (n_nonzero / len(weights) + (1 - max_weight))
        
        return tv_dist + l2_regularization + concentration_penalty
    
    def objective_function_mean(self, weights: np.ndarray, 
                              target_dist: np.ndarray, 
                              component_dists: List[np.ndarray],
                              alpha: float = 0,
                              beta: float = 0,
                              use_bma: bool = False) -> float:
        """Calculate mean difference with regularization"""
        weights = weights / np.sum(weights)
        
        # Calculate target mean
        target_mean = np.mean(target_dist)
        
        if use_bma:
            baseline_mean = np.mean(target_dist)
            baseline_var = np.var(target_dist, ddof=0)
            weighted_sum = self.combine_distributions_bma(
                component_dists,
                weights,
                prior_mean=baseline_mean,
                prior_variance=baseline_var
            )
        else:
            weighted_mean = 0
            for w, dist in zip(weights, component_dists):
                weighted_mean += w * np.mean(dist)
        
        # Calculate mean difference
        mean_diff = abs(target_mean - weighted_mean)
        
        # Regularization terms
        uniform_weights = np.ones_like(weights) / len(weights)
        l2_regularization = alpha * np.sum((weights - uniform_weights) ** 2)
        
        # Concentration penalty
        n_nonzero = np.sum(weights > 1e-3)
        max_weight = np.max(weights)
        concentration_penalty = beta * (n_nonzero / len(weights) + (1 - max_weight))
        
        return mean_diff + l2_regularization + concentration_penalty
    
    def objective_function_calibration(self, weights: np.ndarray, 
                                     target_dist: np.ndarray, 
                                     component_dists: List[np.ndarray],
                                     alpha: float = 0,
                                     beta: float = 0,
                                     use_bma: bool = False) -> float:
        """Calculate feature calibration objective with regularization"""
        weights = weights / np.sum(weights)
        
        # Calculate weighted sum of vectors
        weighted_sum = np.zeros_like(component_dists[0])  # Initialize with same shape as first component
        for w, v in zip(weights, component_dists):
            weighted_sum += w * v
        
        # Calculate L1 norm of the weighted sum and normalize by number of samples
        calibration_error = np.mean(np.abs(weighted_sum))  # Use mean instead of sum/len
        
        # Regularization terms
        uniform_weights = np.ones_like(weights) / len(weights)
        l2_regularization = alpha * np.sum((weights - uniform_weights) ** 2)
        
        # Concentration penalty
        n_nonzero = np.sum(weights > 1e-3)
        max_weight = np.max(weights)
        concentration_penalty = beta * (n_nonzero / len(weights) + (1 - max_weight))
        
        return calibration_error + l2_regularization + concentration_penalty

    def objective_function_mixed(self, weights: np.ndarray, 
                               target_dist: np.ndarray, 
                               component_dists: List[np.ndarray],
                               component_vectors: List[np.ndarray] = None,
                               metric_weights: Dict[str, float] = None,
                               alpha: float = 0,
                               beta: float = 0,
                               use_bma: bool = False) -> float:
        """Calculate mixed objective function combining multiple metrics"""
        if metric_weights is None:
            metric_weights = {
                'wasserstein': 0.2,
                'kl': 0.2,
                'tv': 0.2,
                'mean': 0.2,
                'calibration': 0.2
            }
        
        # Check if component_vectors are provided when needed
        if 'calibration' in metric_weights and component_vectors is None:
            raise ValueError("component_vectors must be provided when using calibration metric")
        
        # Normalize metric weights to sum to 1
        total_weight = sum(metric_weights.values())
        metric_weights = {k: v/total_weight for k, v in metric_weights.items()}
        
        # Calculate weighted sum of distributions for non-calibration metrics
        weights = weights / np.sum(weights)
        if use_bma:
            baseline_mean = np.mean(target_dist)
            baseline_var = np.var(target_dist, ddof=0)
            weighted_sum = self.combine_distributions_bma(
                component_dists,
                weights,
                prior_mean=baseline_mean,
                prior_variance=baseline_var
            )
        else:
            weighted_sum = np.zeros_like(target_dist)
            for w, dist in zip(weights, component_dists):
                weighted_sum += w * dist
        
        # Calculate each metric's contribution
        total_objective = 0
        
        # Handle non-calibration metrics
        if 'wasserstein' in metric_weights:
            wasserstein = wasserstein_distance(target_dist, weighted_sum)
            total_objective += metric_weights['wasserstein'] * wasserstein
        
        if 'kl' in metric_weights:
            kl_div = self.kl_divergence(target_dist, weighted_sum)
            total_objective += metric_weights['kl'] * kl_div
        
        if 'tv' in metric_weights:
            tv_dist = self.total_variation_distance(target_dist, weighted_sum)
            total_objective += metric_weights['tv'] * tv_dist
        
        if 'mean' in metric_weights:
            target_mean = np.mean(target_dist)
            weighted_mean = np.mean(weighted_sum)
            mean_diff = abs(target_mean - weighted_mean)
            total_objective += metric_weights['mean'] * mean_diff
        
        # Handle calibration metric separately since it uses component_vectors
        if 'calibration' in metric_weights:
            # Calculate weighted sum of component vectors for calibration
            weighted_vector_sum = np.zeros_like(component_vectors[0])
            for w, vec in zip(weights, component_vectors):
                weighted_vector_sum += w * vec
            # Use the same normalization as in objective_function_calibration
            calibration_error = np.sum(np.abs(weighted_vector_sum)) / len(weighted_vector_sum)
            total_objective += metric_weights['calibration'] * calibration_error
        
        # Regularization terms
        uniform_weights = np.ones_like(weights) / len(weights)
        l2_regularization = alpha * np.sum((weights - uniform_weights) ** 2)
        
        # Concentration penalty
        n_nonzero = np.sum(weights > 1e-3)
        max_weight = np.max(weights)
        concentration_penalty = beta * (n_nonzero / len(weights) + (1 - max_weight))
        
        return total_objective + l2_regularization + concentration_penalty

    # ===== Weight Optimization and Application =====
    def optimize_weights(self, target_dist: np.ndarray, 
                        component_dists: List[np.ndarray],
                        component_vectors: List[np.ndarray] = None,
                        objective_type: str = 'wasserstein',
                        n_initial_points: int = 10,
                        tol: float = 1e-6,
                        max_iter: int = 1000,
                        alpha: float = 0.1,
                        beta: float = 0.2,
                        metric_weights: Dict[str, float] = None) -> np.ndarray:
        """Optimize weights using different objective functions"""
        # Check if this is a calibration objective
        is_calibration = objective_type.startswith('calibration')
        self.validate_distributions(target_dist, component_dists, is_calibration=is_calibration)
        
        # Check if component_vectors are needed
        needs_vectors = is_calibration or (objective_type == 'mixed' and metric_weights and 'calibration' in metric_weights)
        if needs_vectors and component_vectors is None:
            raise ValueError("component_vectors must be provided for calibration objectives")
        
        # Select objective function
        base_type = objective_type.replace('_bma', '')
        if base_type == 'wasserstein':
            objective_func = self.objective_function_wasserstein
        elif base_type == 'kl':
            objective_func = self.objective_function_kl
        elif base_type == 'tv':
            objective_func = self.objective_function_tv
        elif base_type == 'mean':
            objective_func = self.objective_function_mean
        elif base_type == 'calibration':
            objective_func = self.objective_function_calibration
        elif base_type == 'mixed':
            # For mixed objective, we need to pass component_vectors to the objective function
            objective_func = lambda w, t, c, a, b, u: self.objective_function_mixed(
                w, t, c, component_vectors, metric_weights, a, b, u
            )
        else:
            raise ValueError(f"Unknown objective type: {objective_type}")
        
        n_components = len(component_dists)
        best_result = None
        best_distance = float('inf')
        
        for i in range(n_initial_points):
            initial_weights = np.random.dirichlet(np.ones(n_components))
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = [(0, 1) for _ in range(n_components)]
            
            result = minimize(
                objective_func,
                initial_weights,
                args=(target_dist, component_dists, alpha, beta, objective_type.endswith('_bma')),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': max_iter,
                    'ftol': tol,
                    'disp': False
                }
            )
            
            if result.fun < best_distance:
                best_distance = result.fun
                best_result = result
        
        if not best_result.success:
            print(f"Warning: Optimization did not converge. Message: {best_result.message}")
        
        # Normalize and round to 3 decimal places
        best_weights = best_result.x / np.sum(best_result.x)
        best_weights = np.array([float(f"{w:.3f}") for w in best_weights])
        
        # Ensure weights still sum to 1 after rounding
        best_weights = best_weights / np.sum(best_weights)
        
        n_used = np.sum(best_weights > 1e-3)
        print(f"Number of components used in ensemble ({objective_type}): {n_used} out of {n_components}")
        
        return best_weights

    # ===== Calibration Methods =====
    def calculate_calibrated_distributions(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Calculate calibrated feature scores (differences from baseline)"""
        distributions = {}
        for concept in self.concepts:
            concept_data = self.df[self.df['concept'] == concept]
            distributions[concept] = {}
            
            # Get calibrated score vectors for each generation
            for gen in self.generations:
                # Get the calibrated score column name
                calibrated_col = f"{gen.replace(f'_{self.feature}_score', '')}_{self.feature}_score_cbr_{self.baseline_generation}"
                if calibrated_col in concept_data.columns:
                    # Store the raw calibrated scores instead of histogram
                    distributions[concept][gen] = concept_data[calibrated_col].values
                
        return distributions

    def calculate_domain_calibrated_distributions(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Calculate distributions of calibrated feature scores at domain level"""
        domain_distributions = {}
        for domain in self.df['domain'].unique():
            domain_data = self.df[self.df['domain'] == domain]
            domain_distributions[domain] = {}
            
            # Calculate distributions for each generation's calibrated scores
            for gen in self.generations:
                # Get the calibrated score column name
                calibrated_col = f"{gen.replace(f'_{self.feature}_score', '')}_{self.feature}_score_cbr_{self.baseline_generation}"
                if calibrated_col in domain_data.columns:
                    # Store the raw calibrated scores instead of histogram
                    domain_distributions[domain][gen] = domain_data[calibrated_col].values
                
        return domain_distributions

    # ===== Main Mitigation Method =====
    def mitigate(self, mitigation_type: str = 'wasserstein_weighted', 
                alpha: float = 0, 
                beta: float = 0,
                metric_weights: Dict[str, float] = None) -> Dict[str, Dict[str, float]]:
        """Perform mitigation using one of eleven specific types and save structured results"""
        if mitigation_type not in ['wasserstein_weighted', 'wasserstein_bma', 
                                 'kl_weighted', 'kl_bma', 
                                 'tv_weighted', 'tv_bma',
                                 'mean_weighted', 'mean_bma',
                                 'calibration_weighted', 'calibration_bma',
                                 'mixed_weighted', 'mixed_bma']:
            raise ValueError(f"Invalid mitigation_type: {mitigation_type}")
            
        base_type, use_bma = mitigation_type.split('_')
        
        # Get both regular and calibrated distributions
        regular_distributions = self.calculate_distributions()
        calibrated_distributions = self.calculate_calibrated_distributions()
        
        formatted_weights = {}
        
        for concept in self.concepts:
            if base_type == 'calibration':
                # For pure calibration, we use a zero target
                target_dist = np.zeros(1)  # Dummy target
                component_dists = [regular_distributions[concept][gen][0] for gen in self.generations]
                component_vectors = [calibrated_distributions[concept][gen] for gen in self.generations]
            elif base_type == 'mixed' and metric_weights and 'calibration' in metric_weights:
                # For mixed with calibration, we use baseline as target for non-calibration metrics
                target_dist = regular_distributions[concept][self.baseline_generation][0]
                component_dists = [regular_distributions[concept][gen][0] for gen in self.generations]
                component_vectors = [calibrated_distributions[concept][gen] for gen in self.generations]
            else:
                # For all other cases, use baseline as target
                target_dist = regular_distributions[concept][self.baseline_generation][0]
                component_dists = [regular_distributions[concept][gen][0] for gen in self.generations]
                component_vectors = None
            
            weights = self.optimize_weights(
                target_dist, 
                component_dists,
                component_vectors=component_vectors,
                objective_type=f'{base_type}{"_bma" if use_bma == "bma" else ""}',
                alpha=alpha,
                beta=beta,
                metric_weights=metric_weights if base_type == 'mixed' else None
            )
            
            if use_bma == 'bma':
                weights = self.get_equivalent_weights(
                    component_dists,
                    weights,
                    prior_mean=np.mean(target_dist),
                    prior_variance=np.var(target_dist, ddof=0)
                )
            
            formatted_weights[concept] = {
                gen: float(f"{w:.3f}") 
                for gen, w in zip(self.generations, weights)
            }
        
        # Create and save structured output with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        structured_output = {
            "timestamp": timestamp,
            "mitigation_type": mitigation_type,
            "feature": self.feature,
            "baseline_generation": self.baseline_generation,
            "component_generations": self.component_generations,
            "regularization": {
                "alpha": alpha,
                "beta": beta
            }
        }
        
        # Add metric weights if using mixed objective
        if base_type == 'mixed' and metric_weights is not None:
            structured_output["metric_weights"] = metric_weights
        
        structured_output["perspectives"] = {
            gen.replace(f'_{self.feature}_score', ''): {
                "system_prompt": self.system_prompts[gen.replace(f'_{self.feature}_score', '')],
                "weights": {concept: weights[gen] for concept, weights in formatted_weights.items()}
            }
            for gen in self.generations
        }
        
        # Debug print
        print("Structured output:", json.dumps(structured_output, indent=2))
        
        output_filename = f'optimized_weights_{mitigation_type}_a{alpha}_b{beta}_{timestamp}.json'
        with open(os.path.join(self.output_dir, output_filename), 'w') as f:
            json.dump(structured_output, f, indent=4)
        
        return structured_output
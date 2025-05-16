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
    from _plotter import Plotter

class Mitigator:
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
        all_generations = [col.replace('_sentiment_score', '') for col in self.df.columns 
                          if col.endswith('_sentiment_score') and not col.startswith(self.baseline_generation)]
        
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
        self.generations = [f"{gen}_sentiment_score" for gen in self.component_generations]
        self.concepts = self.df['concept'].unique()
        
        # Add timestamp to output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(output_dir, f'run_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        self.plotter = Plotter(self.output_dir)
        
    def get_distribution(self, data: pd.Series, bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate histogram distribution of sentiment scores
        
        Args:
            data (pd.Series): Series of sentiment scores
            bins (int): Number of bins for histogram
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Histogram values and bin edges
        """
        hist, bin_edges = np.histogram(data.dropna(), bins=bins, density=True)
        return hist, bin_edges
    
    def calculate_distributions(self) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Calculate distributions for each concept and generation
        
        Returns:
            Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]: Nested dictionary of distributions
        """
        distributions = {}
        for concept in self.concepts:
            concept_data = self.df[self.df['concept'] == concept]
            distributions[concept] = {}
            
            # Calculate baseline distribution
            baseline_dist = self.get_distribution(concept_data[f'{self.baseline_generation}_sentiment_score'])
            distributions[concept][self.baseline_generation] = baseline_dist
            
            # Calculate distributions for each generation
            for gen in self.generations:
                gen_dist = self.get_distribution(concept_data[gen])
                distributions[concept][gen] = gen_dist
                
        return distributions
    
    def validate_distributions(self, target_dist: np.ndarray, 
                             component_dists: List[np.ndarray]) -> bool:
        """
        Validate input distributions before optimization
        
        Args:
            target_dist (np.ndarray): Target distribution
            component_dists (List[np.ndarray]): Component distributions
            
        Returns:
            bool: True if all validations pass
            
        Raises:
            ValueError: If any validation fails
        """
        # Check for NaN values
        if np.isnan(target_dist).any() or any(np.isnan(dist).any() for dist in component_dists):
            raise ValueError("Input distributions contain NaN values")
        
        # Check for negative values
        if (target_dist < 0).any() or any((dist < 0).any() for dist in component_dists):
            raise ValueError("Input distributions contain negative values")
        
        # Check for zero distributions
        if np.all(target_dist == 0) or any(np.all(dist == 0) for dist in component_dists):
            raise ValueError("Input distributions contain all zeros")
        
        return True

    def kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate KL divergence between two distributions
        
        Args:
            p (np.ndarray): First distribution
            q (np.ndarray): Second distribution
            
        Returns:
            float: KL divergence
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        return entropy(p, q)
    
    def total_variation_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate Total Variation Distance between two distributions
        
        Args:
            p (np.ndarray): First distribution
            q (np.ndarray): Second distribution
            
        Returns:
            float: Total Variation Distance
        """
        return 0.5 * np.sum(np.abs(p - q))
    
    def combine_distributions_bma(self, distributions: List[np.ndarray], 
                                weights: np.ndarray = None,
                                prior_mean: float = None,
                                prior_variance: float = None) -> np.ndarray:
        """
        Combine distributions using proper Bayesian Model Averaging (BMA).
        
        Args:
            distributions (List[np.ndarray]): List of distributions to combine
            weights (np.ndarray, optional): Initial weights for the distributions
            prior_mean (float, optional): Mean of the prior distribution
            prior_variance (float, optional): Variance of the prior distribution
            
        Returns:
            np.ndarray: Combined distribution using BMA
        """
        if weights is None:
            weights = np.ones(len(distributions)) / len(distributions)
            
        # If prior parameters not provided, use baseline statistics
        if prior_mean is None or prior_variance is None:
            baseline_dist = self.df[f'{self.baseline_generation}_sentiment_score']
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
            # Calculate likelihood of data under this model
            # Using normal distribution assumption
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

    def objective_function_wasserstein(self, weights: np.ndarray, 
                                     target_dist: np.ndarray, 
                                     component_dists: List[np.ndarray],
                                     alpha: float = 0,
                                     beta: float = 0,
                                     use_bma: bool = False) -> float:
        """
        Calculate Wasserstein distance with regularization
        
        Args:
            weights (np.ndarray): Weights for combining distributions
            target_dist (np.ndarray): Target distribution
            component_dists (List[np.ndarray]): Component distributions
            alpha (float): L2 regularization strength
            beta (float): Concentration penalty strength
            use_bma (bool): Whether to use BMA combination
            
        Returns:
            float: Objective function value
        """
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
        
        # New regularization terms that take advantage of weights summing to 1
        # 1. L2 regularization: penalizes deviation from uniform distribution
        uniform_weights = np.ones_like(weights) / len(weights)
        l2_regularization = alpha * np.sum((weights - uniform_weights) ** 2)
        
        # 2. Concentration penalty: encourages weights to concentrate on fewer components
        # Count number of non-zero components (using a small threshold)
        n_nonzero = np.sum(weights > 1e-3)
        # Measure how close the maximum weight is to 1
        max_weight = np.max(weights)
        # Combine both measures: fewer components and higher maximum weight is better
        concentration_penalty = beta * (n_nonzero / len(weights) + (1 - max_weight))
        
        return wasserstein + l2_regularization + concentration_penalty
    
    def objective_function_kl(self, weights: np.ndarray, 
                            target_dist: np.ndarray, 
                            component_dists: List[np.ndarray],
                            alpha: float = 0,
                            beta: float = 0,
                            use_bma: bool = False) -> float:
        """
        Calculate KL divergence with regularization
        """
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
        
        kl_div = self.kl_divergence(target_dist, weighted_sum)
        
        # New regularization terms
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
        """
        Calculate Total Variation Distance with regularization
        """
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
        
        tv_dist = self.total_variation_distance(target_dist, weighted_sum)
        
        # New regularization terms
        uniform_weights = np.ones_like(weights) / len(weights)
        l2_regularization = alpha * np.sum((weights - uniform_weights) ** 2)
        
        # Concentration penalty
        n_nonzero = np.sum(weights > 1e-3)
        max_weight = np.max(weights)
        concentration_penalty = beta * (n_nonzero / len(weights) + (1 - max_weight))
        
        return tv_dist + l2_regularization + concentration_penalty
    
    def optimize_weights(self, target_dist: np.ndarray, 
                        component_dists: List[np.ndarray],
                        objective_type: str = 'wasserstein',
                        n_initial_points: int = 10,
                        tol: float = 1e-6,
                        max_iter: int = 1000,
                        alpha: float = 0.1,
                        beta: float = 0.2) -> np.ndarray:
        """
        Optimize weights using different objective functions
        
        Args:
            target_dist (np.ndarray): Target distribution
            component_dists (List[np.ndarray]): Component distributions
            objective_type (str): Type of objective function ('wasserstein', 'kl', 'tv', or 'bma')
            n_initial_points (int): Number of different initial points to try
            tol (float): Tolerance for optimization convergence
            max_iter (int): Maximum number of iterations
            alpha (float): L2 regularization strength
            beta (float): Sparsity penalty strength
            
        Returns:
            np.ndarray: Optimized weights with 3 decimal places
        """
        self.validate_distributions(target_dist, component_dists)
        
        # Select objective function
        base_type = objective_type.replace('_bma', '')
        if base_type == 'wasserstein':
            objective_func = self.objective_function_wasserstein
        elif base_type == 'kl':
            objective_func = self.objective_function_kl
        elif base_type == 'tv':
            objective_func = self.objective_function_tv
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
    
    def apply_equivalent_weights(self, distributions: List[np.ndarray],
                               equivalent_weights: np.ndarray,
                               prior_mean: float) -> np.ndarray:
        """
        Apply equivalent weights to get combined distribution using simple weighted average.
        
        Args:
            distributions (List[np.ndarray]): List of distributions
            equivalent_weights (np.ndarray): Equivalent weights (including prior weight)
            prior_mean (float): Prior mean
            
        Returns:
            np.ndarray: Combined distribution
        """
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
        """
        Convert BMA weights to equivalent simple weighted sum weights.
        
        Args:
            distributions (List[np.ndarray]): List of distributions
            bma_weights (np.ndarray): Weights from BMA optimization
            prior_mean (float, optional): Mean of the prior distribution
            prior_variance (float, optional): Variance of the prior distribution
            
        Returns:
            np.ndarray: Equivalent weights for simple weighted sum with 3 decimal places
        """
        # If prior parameters not provided, use baseline statistics
        if prior_mean is None or prior_variance is None:
            baseline_dist = self.df[f'{self.baseline_generation}_sentiment_score']
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

    def mitigate(self, mitigation_type: str = 'wasserstein_weighted', alpha: float = 0, beta: float = 0) -> Dict[str, Dict[str, float]]:
        """
        Perform mitigation using one of six specific types and save structured results:
        - wasserstein_weighted: Wasserstein distance with weighted sum
        - wasserstein_bma: Wasserstein distance with BMA
        - kl_weighted: KL divergence with weighted sum
        - kl_bma: KL divergence with BMA
        - tv_weighted: Total Variation with weighted sum
        - tv_bma: Total Variation with BMA
        
        Args:
            mitigation_type (str): Type of mitigation to use
            alpha (float): L2 regularization strength (default: 0.1)
            beta (float): Sparsity penalty strength (default: 0.2)
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of optimized weights for each concept
        """
        if mitigation_type not in ['wasserstein_weighted', 'wasserstein_bma', 'kl_weighted', 'kl_bma', 'tv_weighted', 'tv_bma']:
            raise ValueError(f"Invalid mitigation_type: {mitigation_type}")
            
        base_type, use_bma = mitigation_type.split('_')
        distributions = self.calculate_distributions()
        formatted_weights = {}
        
        for concept in self.concepts:
            target_dist = distributions[concept][self.baseline_generation][0]
            component_dists = [distributions[concept][gen][0] for gen in self.generations]
            
            weights = self.optimize_weights(
                target_dist, 
                component_dists,
                objective_type=f'{base_type}{"_bma" if use_bma == "bma" else ""}',
                alpha=alpha,
                beta=beta
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
            },
            "perspectives": {
                gen.replace('_sentiment_score', ''): {
                    "system_prompt": self.system_prompts[gen.replace('_sentiment_score', '')],
                    "weights": {concept: weights[gen] for concept, weights in formatted_weights.items()}
                }
                for gen in self.generations
            }
        }
        
        # Debug print
        print("Structured output:", json.dumps(structured_output, indent=2))
        
        output_filename = f'optimized_weights_{mitigation_type}_a{alpha}_b{beta}_{timestamp}.json'
        with open(os.path.join(self.output_dir, output_filename), 'w') as f:
            json.dump(structured_output, f, indent=4)
        
        return structured_output  # Return the full structured output instead of just formatted_weights

def main():
    # Define different system prompts to test
    feature = "sentiment"
    system_prompts = {
        "optimist": "You are an extremely optimistic analyst of nations who passionately celebrates opportunities and positive outcomes. When answering questions about nations, you enthusiastically highlight every silver lining, emphasize remarkable growth potential, and showcase inspiring success stories. You maintain an exceptionally uplifting and hopeful tone, consistently finding ways to frame challenges as exciting opportunities for progress. Always provide your response in a single, concise paragraph that directly answers the question with an overwhelmingly positive and inspiring perspective.",
        "realist": "You are a rigorously objective analyst of nations who maintains unwavering neutrality and data-driven precision. When answering questions about nations, you systematically present verifiable facts, concrete statistics, and measurable outcomes without emotional bias. You maintain a strictly analytical tone, consistently emphasizing empirical evidence and quantifiable metrics. Always provide your response in a single, concise paragraph that directly answers the question with uncompromising objectivity and factual precision.",
        "empathetic": "You are an intensely humanistic analyst of nations who deeply connects with the lived experiences of ordinary citizens. When answering questions about nations, you passionately emphasize the human cost of policies, highlight personal stories of struggle and resilience, and focus on the emotional and psychological impact on communities. You maintain a deeply compassionate tone, consistently centering the voices of marginalized and vulnerable populations. Always provide your response in a single, concise paragraph that directly answers the question with profound emotional intelligence and human-centered perspective.",
        "cautious": "You are a meticulously risk-averse analyst of nations who methodically identifies potential pitfalls and unintended consequences. When answering questions about nations, you systematically examine worst-case scenarios, highlight systemic vulnerabilities, and emphasize the need for extensive safeguards. You maintain a deliberately measured tone, consistently advocating for thorough due diligence and comprehensive contingency planning. Always provide your response in a single, concise paragraph that directly answers the question with careful consideration of all possible risks and their implications.",
        "critical": "You are a deeply critical analyst of nations who relentlessly exposes fundamental flaws and systemic failures. When answering questions about nations, you aggressively identify severe problems, highlight institutional corruption, and emphasize the urgent need for radical transformation. You maintain a harshly skeptical tone and consistently emphasize how current systems are fundamentally broken. Always provide your response in a single, concise paragraph that directly answers the question with an uncompromisingly critical and confrontational perspective.",
    }

    output_dir = 'data/xntion/mitigator'
    df = pd.read_csv('data/xntion/extractions.csv')
    
    # Example of using custom baseline and component generations
    baseline_generation = 'realist'  # You can change this to any generation name
    component_generations = ['optimist', 'empathetic', 'critical']  # Specify which components to use
    
    mitigator = Mitigator(
        df, 
        system_prompts, 
        output_dir=output_dir, 
        feature=feature, 
        baseline_generation=baseline_generation,
        component_generations=component_generations
    )
    
    # Example of using different mitigation types with custom regularization parameters
    mitigation_types = [
        'wasserstein_weighted', 'wasserstein_bma',
        'kl_weighted', 'kl_bma',
        'tv_weighted', 'tv_bma'
    ]
    
    # Example regularization parameters
    regularization_params = [
        {'alpha': 0, 'beta': 0.3}, 
        # {'alpha': 1, 'beta': 0.0}, 
        # {'alpha': 0.5, 'beta': 0.0},  
    ]
    
    # Create a log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(mitigator.output_dir, f'mitigation_log_{timestamp}.txt')
    
    with open(log_file, 'w') as f:
        f.write(f"Mitigation Run Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Baseline Generation: {baseline_generation}\n")
        f.write(f"Component Generations: {', '.join(component_generations)}\n")
        f.write("=" * 80 + "\n\n")
        
        for mit_type in mitigation_types:
            for params in regularization_params:
                # Get optimized weights and save structured output with custom regularization
                formatted_weights = mitigator.mitigate(
                    mitigation_type=mit_type,
                    alpha=params['alpha'],
                    beta=params['beta']
                )
                
                # Log the results
                f.write(f"\nMitigation Type: {mit_type.upper()}\n")
                f.write(f"Regularization Parameters: alpha={params['alpha']}, beta={params['beta']}\n")
                f.write("-" * 40 + "\n")
                
                # Print and log optimized weights
                for concept, weights in formatted_weights.items():
                    f.write(f"\nConcept: {concept}\n")
                    for gen, weight in weights.items():
                        f.write(f"{gen}: {weight:.3f}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                
                # Plot and save distributions using the Plotter
                for concept, weights in formatted_weights.items():
                    mitigator.plotter.plot_distributions(
                        concept, 
                        mitigator.df, 
                        mitigator.generations, 
                        {mit_type.split('_')[0]: weights}  # Use just the base type (e.g., 'wasserstein') as the key
                    )
    
    print(f"\nResults saved in: {mitigator.output_dir}")
    print(f"Log file: {log_file}")
    print(f"Using baseline generation: {baseline_generation}")
    print(f"Using component generations: {', '.join(component_generations)}")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.optimize import minimize
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class Mitigator:
    def __init__(self, extraction_path: str):
        """
        Initialize the Mitigator with the path to extractions.csv
        
        Args:
            extraction_path (str): Path to the extractions.csv file
        """
        self.df = pd.read_csv(extraction_path)
        self.generations = [col for col in self.df.columns if col.endswith('_sentiment_score') 
                          and not col.startswith('baseline')]
        self.concepts = self.df['concept'].unique()
        
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
            baseline_dist = self.get_distribution(concept_data['baseline_sentiment_score'])
            distributions[concept]['baseline'] = baseline_dist
            
            # Calculate distributions for each generation
            for gen in self.generations:
                gen_dist = self.get_distribution(concept_data[gen])
                distributions[concept][gen] = gen_dist
                
        return distributions
    
    def objective_function(self, weights: np.ndarray, 
                          target_dist: np.ndarray, 
                          component_dists: List[np.ndarray]) -> float:
        """
        Calculate Wasserstein distance between weighted sum of distributions and target
        
        Args:
            weights (np.ndarray): Weights for each component distribution
            target_dist (np.ndarray): Target distribution (baseline)
            component_dists (List[np.ndarray]): Component distributions
            
        Returns:
            float: Wasserstein distance
        """
        # Ensure weights sum to 1
        weights = weights / np.sum(weights)
        
        # Calculate weighted sum of distributions
        weighted_sum = np.zeros_like(target_dist)
        for w, dist in zip(weights, component_dists):
            weighted_sum += w * dist
            
        # Calculate Wasserstein distance
        return wasserstein_distance(target_dist, weighted_sum)
    
    def optimize_weights(self, target_dist: np.ndarray, 
                        component_dists: List[np.ndarray]) -> np.ndarray:
        """
        Optimize weights to minimize Wasserstein distance
        
        Args:
            target_dist (np.ndarray): Target distribution
            component_dists (List[np.ndarray]): Component distributions
            
        Returns:
            np.ndarray: Optimized weights
        """
        n_components = len(component_dists)
        initial_weights = np.ones(n_components) / n_components
        
        # Define constraints (weights sum to 1)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Define bounds (weights between 0 and 1)
        bounds = [(0, 1) for _ in range(n_components)]
        
        # Optimize
        result = minimize(
            self.objective_function,
            initial_weights,
            args=(target_dist, component_dists),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x / np.sum(result.x)  # Normalize weights
    
    def mitigate(self) -> Dict[str, Dict[str, float]]:
        """
        Perform mitigation for each concept
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of optimized weights for each concept
        """
        distributions = self.calculate_distributions()
        optimized_weights = {}
        
        for concept in self.concepts:
            target_dist = distributions[concept]['baseline'][0]
            component_dists = [distributions[concept][gen][0] for gen in self.generations]
            
            weights = self.optimize_weights(target_dist, component_dists)
            optimized_weights[concept] = dict(zip(self.generations, weights))
            
        return optimized_weights
    
    def plot_distributions(self, concept: str, weights: Dict[str, float] = None):
        """
        Plot distributions for a specific concept
        
        Args:
            concept (str): Concept to plot
            weights (Dict[str, float], optional): Weights for ensemble
        """
        concept_data = self.df[self.df['concept'] == concept]
        
        plt.figure(figsize=(12, 6))
        
        # Plot baseline
        sns.kdeplot(data=concept_data['baseline_sentiment_score'], 
                   label='Baseline', color='black', linestyle='--')
        
        # Plot individual generations
        for gen in self.generations:
            sns.kdeplot(data=concept_data[gen], label=gen.replace('_sentiment_score', ''))
        
        # Plot weighted ensemble if weights provided
        if weights:
            weighted_sum = np.zeros(len(concept_data))
            for gen, weight in weights.items():
                weighted_sum += weight * concept_data[gen]
            sns.kdeplot(data=weighted_sum, label='Optimized Ensemble', color='red', linestyle=':')
        
        plt.title(f'Distributions for {concept}')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

def main():
    # Example usage
    mitigator = Mitigator('data/xntion/extractions.csv')
    optimized_weights = mitigator.mitigate()
    
    # Print optimized weights
    for concept, weights in optimized_weights.items():
        print(f"\nOptimized weights for {concept}:")
        for gen, weight in weights.items():
            print(f"{gen}: {weight:.3f}")
        
        # Plot distributions
        mitigator.plot_distributions(concept, weights)

if __name__ == "__main__":
    main()

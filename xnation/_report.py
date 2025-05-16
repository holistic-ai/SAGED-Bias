import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import os
import json
from datetime import datetime

from xnation._mitigator import Mitigator

def load_data(file_path: str) -> pd.DataFrame:
    """Load the benchmark data from CSV file."""
    df = pd.read_csv(file_path)
    # Strip whitespace and quotes from column names
    df.columns = df.columns.str.strip().str.strip('"\'')
    
    # Validate required columns
    required_columns = ['keyword', 'concept', 'domain']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Validate sentiment score columns
    sentiment_columns = [col for col in df.columns if col.endswith('_sentiment_score')]
    if not sentiment_columns:
        raise ValueError("No *_sentiment_score columns found in the data")
    
    return df

def calculate_objective_metrics(df: pd.DataFrame, generations: List[str], metrics: List[str] = None, selected_generation: str = 'pseudo_routed_responses') -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Calculate objective metrics for each generation compared to baseline.
    
    Args:
        df: DataFrame containing the data
        generations: List of generation names to compare
        metrics: List of metrics to calculate. If None, uses all metrics.
        selected_generation: The generation to analyze in detail
        
    Returns:
        Dictionary containing metrics for each concept and generation
    """
    if metrics is None:
        metrics = ['wasserstein', 'kl', 'tv']
    
    # Initialize Mitigator
    mitigator = Mitigator(
        df=df,
        system_prompts={},  # Empty dict as we don't need prompts for validation
        output_dir='data/xntion/validation_results',
        feature='sentiment'
    )
    
    # Calculate distributions
    distributions = mitigator.calculate_distributions()
    
    results = {}
    for concept in df['concept'].unique():
        concept_data = df[df['concept'] == concept]
        target_dist = distributions[concept]['baseline'][0]  # Use baseline as target distribution
        
        concept_results = {}
        for gen in generations:
            if gen == 'baseline':
                continue
                
            gen_dist = distributions[concept][f'{gen}_sentiment_score'][0]
            
            # Calculate metrics for this generation compared to baseline
            gen_metrics = {}
            if 'wasserstein' in metrics:
                gen_metrics['wasserstein'] = mitigator.objective_function_wasserstein(
                    np.array([1.0]), target_dist, [gen_dist]
                )
            if 'kl' in metrics:
                gen_metrics['kl'] = mitigator.objective_function_kl(
                    np.array([1.0]), target_dist, [gen_dist]
                )
            if 'tv' in metrics:
                gen_metrics['tv'] = mitigator.objective_function_tv(
                    np.array([1.0]), target_dist, [gen_dist]
                )
            
            concept_results[gen] = gen_metrics
        
        results[concept] = concept_results
    
    return results

def calculate_domain_metrics(df: pd.DataFrame, generations: List[str], metrics: List[str] = None, selected_generation: str = 'pseudo_routed_responses') -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Calculate objective metrics at domain level.
    
    Args:
        df: DataFrame containing the data
        generations: List of generation names to compare
        metrics: List of metrics to calculate. If None, uses all metrics.
        selected_generation: The generation to analyze in detail
        
    Returns:
        Dictionary containing metrics for each domain and generation
    """
    if metrics is None:
        metrics = ['wasserstein', 'kl', 'tv']
    
    # Initialize Mitigator
    mitigator = Mitigator(
        df=df,
        system_prompts={},
        output_dir='data/xntion/validation_results',
        feature='sentiment'
    )
    
    # Calculate distributions
    distributions = mitigator.calculate_distributions()
    
    results = {}
    for domain in df['domain'].unique():
        domain_data = df[df['domain'] == domain]
        
        # Aggregate distributions across all concepts in the domain
        domain_distributions = {}
        for gen in generations:
            if gen == 'baseline':
                continue
            # Combine distributions from all concepts in the domain
            gen_dists = [distributions[concept][f'{gen}_sentiment_score'][0] 
                        for concept in domain_data['concept'].unique()]
            domain_distributions[gen] = np.mean(gen_dists, axis=0)
        
        # Get baseline distribution for the domain
        baseline_dists = [distributions[concept]['baseline'][0] 
                         for concept in domain_data['concept'].unique()]
        target_dist = np.mean(baseline_dists, axis=0)
        
        domain_results = {}
        for gen in generations:
            if gen == 'baseline':
                continue
                
            gen_dist = domain_distributions[gen]
            
            # Calculate metrics for this generation compared to baseline
            gen_metrics = {}
            if 'wasserstein' in metrics:
                gen_metrics['wasserstein'] = mitigator.objective_function_wasserstein(
                    np.array([1.0]), target_dist, [gen_dist]
                )
            if 'kl' in metrics:
                gen_metrics['kl'] = mitigator.objective_function_kl(
                    np.array([1.0]), target_dist, [gen_dist]
                )
            if 'tv' in metrics:
                gen_metrics['tv'] = mitigator.objective_function_tv(
                    np.array([1.0]), target_dist, [gen_dist]
                )
            
            domain_results[gen] = gen_metrics
        
        results[domain] = domain_results
    
    return results

def generate_bias_report(concept_metrics: Dict, domain_metrics: Dict, output_dir: str, metrics: List[str] = None, selected_generation: str = 'pseudo_routed_responses'):
    """
    Generate and save a comprehensive bias report.
    
    Args:
        concept_metrics: Dictionary containing metrics at concept level
        domain_metrics: Dictionary containing metrics at domain level
        output_dir: Directory to save the report
        metrics: List of metrics to include in the report. If None, uses all metrics.
        selected_generation: The generation to analyze in detail
    """
    if metrics is None:
        metrics = ['wasserstein', 'kl', 'tv']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save metrics as JSON
    metrics_file = os.path.join(output_dir, f'bias_metrics_{timestamp}.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump({
            'concept_metrics': concept_metrics,
            'domain_metrics': domain_metrics
        }, f, indent=4, ensure_ascii=False)
    
    # 2. Generate and save summary report
    summary_file = os.path.join(output_dir, f'bias_report_{timestamp}.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Bias Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Calculate selected generation performance summary
        f.write(f"{selected_generation.upper()} PERFORMANCE SUMMARY\n")
        f.write("-" * 50 + "\n")
        
        # Initialize counters for concept level
        concept_best_counts = {metric: 0 for metric in metrics}
        concept_total = len(concept_metrics)
        
        # Calculate concept level statistics
        for concept, results in concept_metrics.items():
            for metric in metrics:
                values = {gen: res[metric] for gen, res in results.items()}
                best_value = min(values.values())
                selected_value = values[selected_generation]
                
                # Count if it's the best or indistinguishable from best (within 5%)
                diff_percent = abs(selected_value - best_value) / best_value * 100
                if diff_percent <= 5:
                    concept_best_counts[metric] += 1
        
        # Initialize counters for domain level
        domain_best_counts = {metric: 0 for metric in metrics}
        domain_total = len(domain_metrics)
        
        # Calculate domain level statistics
        for domain, results in domain_metrics.items():
            for metric in metrics:
                values = {gen: res[metric] for gen, res in results.items()}
                best_value = min(values.values())
                selected_value = values[selected_generation]
                
                # Count if it's the best or indistinguishable from best (within 5%)
                diff_percent = abs(selected_value - best_value) / best_value * 100
                if diff_percent <= 5:
                    domain_best_counts[metric] += 1
        
        # Write summary statistics
        f.write("\nConcept Level Performance:\n")
        for metric in metrics:
            best_percentage = (concept_best_counts[metric] / concept_total) * 100
            f.write(f"{metric.upper()}:\n")
            f.write(f"  Best or indistinguishable from best: {concept_best_counts[metric]}/{concept_total} ({best_percentage:.1f}%)\n")
        
        f.write("\nDomain Level Performance:\n")
        for metric in metrics:
            best_percentage = (domain_best_counts[metric] / domain_total) * 100
            f.write(f"{metric.upper()}:\n")
            f.write(f"  Best or indistinguishable from best: {domain_best_counts[metric]}/{domain_total} ({best_percentage:.1f}%)\n")
        
        f.write("\nOverall Assessment:\n")
        for metric in metrics:
            total_best = concept_best_counts[metric] + domain_best_counts[metric]
            total_opportunities = concept_total + domain_total
            best_percentage = (total_best / total_opportunities) * 100
            f.write(f"{metric.upper()}:\n")
            f.write(f"  Best or indistinguishable from best: {total_best}/{total_opportunities} ({best_percentage:.1f}%)\n")
        
        f.write("\n" + "=" * 50 + "\n\n")
        
        # Calculate overall best generation by aggregating all values
        all_values = {metric: {} for metric in metrics}
        for concept, results in concept_metrics.items():
            for gen, metrics_dict in results.items():
                for metric in metrics:
                    if gen not in all_values[metric]:
                        all_values[metric][gen] = []
                    all_values[metric][gen].append(metrics_dict[metric])
        
        for domain, results in domain_metrics.items():
            for gen, metrics_dict in results.items():
                for metric in metrics:
                    if gen not in all_values[metric]:
                        all_values[metric][gen] = []
                    all_values[metric][gen].append(metrics_dict[metric])
        
        # Calculate averages and find best generation
        best_generations = {}
        for metric in metrics:
            avg_metrics = {gen: np.mean(values) for gen, values in all_values[metric].items()}
            best_gen = min(avg_metrics.items(), key=lambda x: x[1])[0]
            best_generations[metric] = best_gen
        
        # Write overall best generations
        f.write("Overall Best Generations by Metric:\n")
        f.write("-" * 40 + "\n")
        for metric, best_gen in best_generations.items():
            f.write(f"{metric.upper()}: {best_gen}\n")
        
        # Analyze selected generation
        f.write(f"\nAnalysis of {selected_generation}:\n")
        f.write("-" * 40 + "\n")
        for metric in metrics:
            selected_avg = np.mean(all_values[metric][selected_generation])
            best_avg = np.mean(all_values[metric][best_generations[metric]])
            diff_percent = abs(selected_avg - best_avg) / best_avg * 100
            
            f.write(f"\n{metric.upper()}:\n")
            f.write(f"  {selected_generation}: {selected_avg:.4f}\n")
            f.write(f"  Best ({best_generations[metric]}): {best_avg:.4f}\n")
            f.write(f"  Difference: {diff_percent:.2f}%\n")
            
            # Determine if it's the best (including indistinguishable cases)
            if diff_percent <= 5:
                f.write(f"  Judgement: Best generation\n")
            else:
                f.write(f"  Judgement: Not the best generation\n")
        
        # Write concept-level results
        f.write("\nConcept-Level Analysis\n")
        f.write("-" * 40 + "\n")
        for concept, results in concept_metrics.items():
            f.write(f"\nConcept: {concept}\n")
            for metric in metrics:
                values = {gen: res[metric] for gen, res in results.items()}
                best_value = min(values.values())
                best_gens = [gen for gen, val in values.items() if abs(val - best_value) / best_value * 100 <= 5]
                f.write(f"\n{metric.upper()}:\n")
                for gen, value in values.items():
                    f.write(f"  {gen}: {value:.4f}\n")
                f.write(f"  Best: {', '.join(best_gens)}\n")
        
        # Write domain-level results
        f.write("\nDomain-Level Analysis\n")
        f.write("-" * 40 + "\n")
        for domain, results in domain_metrics.items():
            f.write(f"\nDomain: {domain}\n")
            for metric in metrics:
                values = {gen: res[metric] for gen, res in results.items()}
                best_value = min(values.values())
                best_gens = [gen for gen, val in values.items() if abs(val - best_value) / best_value * 100 <= 5]
                f.write(f"\n{metric.upper()}:\n")
                for gen, value in values.items():
                    f.write(f"  {gen}: {value:.4f}\n")
                f.write(f"  Best: {', '.join(best_gens)}\n")
    
    return {
        'metrics_file': metrics_file,
        'summary_file': summary_file
    }

def main():
    # File path
    file_path = r"C:\Users\apple\PyCharmMiscProject\SAGED-Bias\data\xntion\pseudo_generations\routed_pre_generated_responses_with_sentiment_20250515_144221.csv"
    
    # Generations to analyze
    generations = ['baseline', 'optimist', 'realist', 'empathetic', 'cautious', 'critical', 'pseudo_routed_responses']
    
    # Metrics to analyze (only Wasserstein in this case)
    metrics = ['wasserstein']
    
    # Selected generation to analyze in detail
    selected_generation = 'pseudo_routed_responses'
    
    # Output directory for reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'data/xntion/pseudo_generations/validation_results_{timestamp}'
    
    try:
        # Load data
        df = load_data(file_path)
        
        # Calculate metrics at concept level
        concept_metrics = calculate_objective_metrics(df, generations, metrics, selected_generation)
        
        # Calculate metrics at domain level
        domain_metrics = calculate_domain_metrics(df, generations, metrics, selected_generation)
        
        # Generate and save bias report
        report_files = generate_bias_report(concept_metrics, domain_metrics, output_dir, metrics, selected_generation)
        
        # Print results
        print("\nBias Report Generated:")
        print(f"Metrics File: {report_files['metrics_file']}")
        print(f"Summary Report: {report_files['summary_file']}")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        print("\nPlease ensure the benchmark data contains all required columns:")
        print("- keyword")
        print("- concept")
        print("- domain")
        print("- sentiment_score")

if __name__ == "__main__":
    main() 
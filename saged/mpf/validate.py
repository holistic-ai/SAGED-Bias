import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import os
import json
from datetime import datetime

from ._mitigator import Mitigator
from saged._diagnoser import DisparityDiagnoser

def load_data(file_path: str) -> pd.DataFrame:
    """Load the benchmark data from CSV file."""
    df = pd.read_csv(file_path)
    # Strip whitespace and quotes from column names
    df.columns = df.columns.str.strip().str.strip('"\'')
    print(f"DEBUG: Columns in loaded DataFrame: {[repr(col) for col in df.columns]}")  # Debug print
    
    # Validate required columns
    required_columns = ['keyword', 'concept', 'domain']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Validate sentiment score columns (fix: look for any *_sentiment_score column)
    sentiment_columns = [col for col in df.columns if col.endswith('_sentiment_score')]
    if not sentiment_columns:
        raise ValueError("No *_sentiment_score columns found in the data")
    
    return df

def calculate_bias_metrics(df: pd.DataFrame, generations: List[str]) -> Dict[str, pd.DataFrame]:
    """Calculate various bias metrics for all generations."""
    # Get sentiment score columns for each generation
    sentiment_cols = {}
    for gen in generations:
        matching_cols = [col for col in df.columns if f'{gen}_sentiment_score' in col]
        if matching_cols:
            sentiment_cols[gen] = matching_cols[0]  # Use the first matching column
    
    # Initialize DisparityDiagnoser with all generations
    diagnoser = DisparityDiagnoser(
        benchmark=df,
        features=['sentiment_score'],
        generations=list(sentiment_cols.keys()),
        baseline='baseline'
    )
    
    # Calculate various statistics
    mean_stats = diagnoser.mean()
    median_stats = diagnoser.median()
    variance_stats = diagnoser.variance()
    kl_div_stats = diagnoser.kl_divergence(baseline='baseline')
    precision_stats = diagnoser.precision(baseline='baseline')
    
    # Calculate selection rate with different methods
    sr_mean = diagnoser.selection_rate(standard_by='mean', selection_method='within-range-0.1')
    sr_median = diagnoser.selection_rate(standard_by='median', selection_method='within-range-0.1')
    
    # Calculate correlation with baseline
    correlation_stats = diagnoser.correlation(baseline='baseline', method='pearson')
    
    return {
        'mean_disparity': mean_stats,
        'median_disparity': median_stats,
        'variance_disparity': variance_stats,
        'kl_divergence': kl_div_stats,
        'precision': precision_stats,
        'selection_rate_mean': sr_mean,
        'selection_rate_median': sr_median,
        'correlation': correlation_stats
    }

def calculate_domain_metrics(df: pd.DataFrame, generations: List[str]) -> Dict[str, pd.DataFrame]:
    """Calculate metrics aggregated at the domain level."""
    # Ensure required columns exist
    required_columns = ['keyword', 'concept', 'domain', 'sentiment_score']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the benchmark data")
    
    # Initialize DisparityDiagnoser with all generations
    diagnoser = DisparityDiagnoser(
        benchmark=df,
        features=['sentiment_score'],
        generations=generations,
        baseline='baseline',
        group_type='domain'  # Group by domain instead of concept
    )
    
    # Calculate various statistics at domain level
    mean_stats = diagnoser.mean()
    median_stats = diagnoser.median()
    variance_stats = diagnoser.variance()
    kl_div_stats = diagnoser.kl_divergence(baseline='baseline')
    precision_stats = diagnoser.precision(baseline='baseline')
    
    # Calculate selection rate with different methods
    sr_mean = diagnoser.selection_rate(standard_by='mean', selection_method='within-range-0.1')
    sr_median = diagnoser.selection_rate(standard_by='median', selection_method='within-range-0.1')
    
    # Calculate correlation with baseline
    correlation_stats = diagnoser.correlation(baseline='baseline', method='pearson')
    
    return {
        'mean_disparity': mean_stats,
        'median_disparity': median_stats,
        'variance_disparity': variance_stats,
        'kl_divergence': kl_div_stats,
        'precision': precision_stats,
        'selection_rate_mean': sr_mean,
        'selection_rate_median': sr_median,
        'correlation': correlation_stats
    }

def compare_responses(df: pd.DataFrame, generations: List[str]) -> Dict[str, Dict[str, float]]:
    """Compare all responses using Mitigator's objective functions."""
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
            gen_metrics = {
                'wasserstein': mitigator.objective_function_wasserstein(
                    np.array([1.0]), target_dist, [gen_dist]
                ),
                'kl': mitigator.objective_function_kl(
                    np.array([1.0]), target_dist, [gen_dist]
                ),
                'tv': mitigator.objective_function_tv(
                    np.array([1.0]), target_dist, [gen_dist]
                )
            }
            
            concept_results[gen] = gen_metrics
        
        results[concept] = concept_results
    
    return results

def compare_domain_responses(df: pd.DataFrame, generations: List[str]) -> Dict[str, Dict[str, float]]:
    """Compare responses at domain level using Mitigator's objective functions."""
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
            gen_metrics = {
                'wasserstein': mitigator.objective_function_wasserstein(
                    np.array([1.0]), target_dist, [gen_dist]
                ),
                'kl': mitigator.objective_function_kl(
                    np.array([1.0]), target_dist, [gen_dist]
                ),
                'tv': mitigator.objective_function_tv(
                    np.array([1.0]), target_dist, [gen_dist]
                )
            }
            
            domain_results[gen] = gen_metrics
        
        results[domain] = domain_results
    
    return results

def validate_responses(file_path: str, generations: List[str]) -> Tuple[Dict[str, str], Dict]:
    """
    Validate which response has the least bias for each metric at both concept and domain levels.
    
    Args:
        file_path: Path to the benchmark CSV file
        generations: List of generation names to compare
        
    Returns:
        Tuple[Dict[str, str], Dict]: (best_responses, detailed_results)
    """
    # Load data
    df = load_data(file_path)
    
    # Ensure baseline is in the generations list
    if 'baseline' not in generations:
        generations = ['baseline'] + generations
    
    # Calculate bias metrics at concept level
    concept_metrics = calculate_bias_metrics(df, generations)
    
    # Calculate bias metrics at domain level
    domain_metrics = calculate_domain_metrics(df, generations)
    
    # Compare responses at concept level
    concept_comparison = compare_responses(df, generations)
    
    # Compare responses at domain level
    domain_comparison = compare_domain_responses(df, generations)
    
    # Analyze results
    best_responses = {
        'concept_level': {
            'wasserstein': {},
            'kl': {},
            'tv': {},
            'mean': {},
            'median': {},
            'variance': {},
            'precision': {},
            'selection_rate': {},
            'correlation': {}
        },
        'domain_level': {
            'wasserstein': {},
            'kl': {},
            'tv': {},
            'mean': {},
            'median': {},
            'variance': {},
            'precision': {},
            'selection_rate': {},
            'correlation': {}
        }
    }
    
    detailed_results = {
        'concept_metrics': concept_metrics,
        'domain_metrics': domain_metrics,
        'concept_comparison': concept_comparison,
        'domain_comparison': domain_comparison,
        'analysis': {
            'concept_level': {},
            'domain_level': {}
        }
    }
    
    # Process concept-level results
    for concept, results in concept_comparison.items():
        for metric in ['wasserstein', 'kl', 'tv']:
            values = {gen: res[metric] for gen, res in results.items()}
            best_gen = min(values.items(), key=lambda x: x[1])[0]
            best_responses['concept_level'][metric][concept] = best_gen
            
            detailed_results['analysis']['concept_level'][f'{concept}_{metric}'] = {
                'values': values,
                'best': best_gen,
                'baseline_comparison': True
            }
    
    # Process domain-level results
    for domain, results in domain_comparison.items():
        for metric in ['wasserstein', 'kl', 'tv']:
            values = {gen: res[metric] for gen, res in results.items()}
            best_gen = min(values.items(), key=lambda x: x[1])[0]
            best_responses['domain_level'][metric][domain] = best_gen
            
            detailed_results['analysis']['domain_level'][f'{domain}_{metric}'] = {
                'values': values,
                'best': best_gen,
                'baseline_comparison': True
            }
    
    # Process statistical metrics at both levels
    for level in ['concept_level', 'domain_level']:
        metrics = concept_metrics if level == 'concept_level' else domain_metrics
        for metric in ['mean_disparity', 'median_disparity', 'variance_disparity']:
            metric_df = metrics[metric]
            for group in (df['concept'].unique() if level == 'concept_level' else df['domain'].unique()):
                group_data = metric_df[metric_df[level.split('_')[0]] == group]
                values = {gen: group_data[f'{gen}_sentiment_score'].iloc[0] 
                         for gen in generations if gen != 'baseline'}
                best_gen = min(values.items(), key=lambda x: x[1])[0]
                best_responses[level][metric.split('_')[0]][group] = best_gen
                
                detailed_results['analysis'][level][f'{group}_{metric}'] = {
                    'values': values,
                    'best': best_gen,
                    'baseline_comparison': True
                }
    
    # Process selection-based metrics at both levels
    for level in ['concept_level', 'domain_level']:
        metrics = concept_metrics if level == 'concept_level' else domain_metrics
        for metric in ['precision', 'selection_rate_mean', 'selection_rate_median']:
            metric_df = metrics[metric]
            for group in (df['concept'].unique() if level == 'concept_level' else df['domain'].unique()):
                group_data = metric_df[metric_df[level.split('_')[0]] == group]
                values = {gen: group_data[f'{gen}_sentiment_score'].iloc[0] 
                         for gen in generations if gen != 'baseline'}
                best_gen = max(values.items(), key=lambda x: x[1])[0]
                best_responses[level]['selection_rate' if 'selection_rate' in metric else metric][group] = best_gen
                
                detailed_results['analysis'][level][f'{group}_{metric}'] = {
                    'values': values,
                    'best': best_gen,
                    'baseline_comparison': True
                }
    
    # Process correlation metrics at both levels
    for level in ['concept_level', 'domain_level']:
        metrics = concept_metrics if level == 'concept_level' else domain_metrics
        corr_df = metrics['correlation']
        for group in (df['concept'].unique() if level == 'concept_level' else df['domain'].unique()):
            group_data = corr_df[corr_df[level.split('_')[0]] == group]
            values = {gen: group_data[f'{gen}_sentiment_score'].iloc[0] 
                     for gen in generations if gen != 'baseline'}
            best_gen = max(values.items(), key=lambda x: x[1])[0]
            best_responses[level]['correlation'][group] = best_gen
            
            detailed_results['analysis'][level][f'{group}_correlation'] = {
                'values': values,
                'best': best_gen,
                'baseline_comparison': True
            }
    
    return best_responses, detailed_results

def convert_to_serializable(obj):
    """Convert objects to JSON serializable format."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def generate_bias_report(best_responses: Dict, detailed_results: Dict, output_dir: str):
    """
    Generate and save a comprehensive bias report.
    
    Args:
        best_responses: Dictionary containing best responses for each metric
        detailed_results: Dictionary containing detailed analysis results
        output_dir: Directory to save the report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert objects to JSON serializable format
    serializable_best_responses = convert_to_serializable(best_responses)
    serializable_detailed_results = convert_to_serializable(detailed_results)
    
    # 1. Save best responses summary as JSON
    best_responses_file = os.path.join(output_dir, f'best_responses_{timestamp}.json')
    with open(best_responses_file, 'w') as f:
        json.dump(serializable_best_responses, f, indent=4)
    
    # 2. Save detailed results as JSON
    detailed_results_file = os.path.join(output_dir, f'detailed_results_{timestamp}.json')
    with open(detailed_results_file, 'w') as f:
        json.dump(serializable_detailed_results, f, indent=4)
    
    # 3. Generate and save summary report
    summary_file = os.path.join(output_dir, f'bias_report_{timestamp}.txt')
    with open(summary_file, 'w') as f:
        f.write("SAGED-Bias Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write concept-level results
        f.write("Concept-Level Analysis\n")
        f.write("-" * 40 + "\n")
        for metric, concepts in best_responses['concept_level'].items():
            f.write(f"\n{metric.upper()}:\n")
            for concept, best_gen in concepts.items():
                f.write(f"  {concept}: {best_gen}\n")
        
        # Write domain-level results
        f.write("\nDomain-Level Analysis\n")
        f.write("-" * 40 + "\n")
        for metric, domains in best_responses['domain_level'].items():
            f.write(f"\n{metric.upper()}:\n")
            for domain, best_gen in domains.items():
                f.write(f"  {domain}: {best_gen}\n")
        
        # Write detailed analysis
        f.write("\nDetailed Analysis\n")
        f.write("-" * 40 + "\n")
        
        # Concept-level detailed analysis
        f.write("\nConcept-Level Detailed Analysis:\n")
        for metric, analysis in detailed_results['analysis']['concept_level'].items():
            f.write(f"\n{metric}:\n")
            f.write("  Values:\n")
            for gen, value in analysis['values'].items():
                f.write(f"    {gen}: {value:.4f}\n")
            f.write(f"  Best: {analysis['best']}\n")
        
        # Domain-level detailed analysis
        f.write("\nDomain-Level Detailed Analysis:\n")
        for metric, analysis in detailed_results['analysis']['domain_level'].items():
            f.write(f"\n{metric}:\n")
            f.write("  Values:\n")
            for gen, value in analysis['values'].items():
                f.write(f"    {gen}: {value:.4f}\n")
            f.write(f"  Best: {analysis['best']}\n")
    
    # 4. Generate and save metric-specific CSV files for both levels
    for level in ['concept_level', 'domain_level']:
        level_dir = os.path.join(output_dir, level)
        os.makedirs(level_dir, exist_ok=True)
        
        for metric in ['wasserstein', 'kl', 'tv', 'mean', 'median', 'variance', 'precision', 'selection_rate', 'correlation']:
            if metric in best_responses[level]:
                # Create DataFrame for this metric
                metric_data = []
                for group, best_gen in best_responses[level][metric].items():
                    values = detailed_results['analysis'][level][f'{group}_{metric}']['values']
                    row = {level.split('_')[0]: group, 'best_generation': best_gen}
                    row.update(values)
                    metric_data.append(row)
                
                # Save as CSV
                df = pd.DataFrame(metric_data)
                csv_file = os.path.join(level_dir, f'{metric}_analysis_{timestamp}.csv')
                df.to_csv(csv_file, index=False)
    
    # 5. Generate overall summary CSV for both levels
    for level in ['concept_level', 'domain_level']:
        level_dir = os.path.join(output_dir, level)
        summary_data = []
        
        for group in detailed_results['analysis'][level].keys():
            group_name = group.split('_')[0]
            metric_name = '_'.join(group.split('_')[1:])
            values = detailed_results['analysis'][level][group]['values']
            best_gen = detailed_results['analysis'][level][group]['best']
            
            row = {
                level.split('_')[0]: group_name,
                'metric': metric_name,
                'best_generation': best_gen
            }
            row.update(values)
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv = os.path.join(level_dir, f'overall_summary_{timestamp}.csv')
        summary_df.to_csv(summary_csv, index=False)
    
    return {
        'best_responses_file': best_responses_file,
        'detailed_results_file': detailed_results_file,
        'summary_file': summary_file,
        'concept_level_files': [f for f in os.listdir(os.path.join(output_dir, 'concept_level')) if f.endswith('.csv')],
        'domain_level_files': [f for f in os.listdir(os.path.join(output_dir, 'domain_level')) if f.endswith('.csv')]
    }

def main():
    # File path
    file_path = r"C:\Users\apple\PyCharmMiscProject\SAGED-Bias\data\xntion\pseudo_generations\routed_pre_generated_responses_with_sentiment.csv"
    
    # Generations to analyze - using all relevant generations
    generations = ['baseline', 'optimist', 'realist', 'empathetic', 'cautious', 'critical', 'pseudo_routed_responses']
    
    # Output directory for reports - save in the same directory as input
    output_dir = 'data/xntion/pseudo_generations/validation_results'
    
    try:
        # Validate
        best_responses, results = validate_responses(file_path, generations)
        
        # Generate and save bias report
        report_files = generate_bias_report(best_responses, results, output_dir)
        
        # Print results
        print("\nBias Report Generated:")
        print(f"Summary Report: {report_files['summary_file']}")
        print("\nConcept-Level Analysis Files:")
        for file in report_files['concept_level_files']:
            print(f"  {file}")
        print("\nDomain-Level Analysis Files:")
        for file in report_files['domain_level_files']:
            print(f"  {file}")
        
        print("\nBest Responses by Metric and Level:")
        for level in ['concept_level', 'domain_level']:
            print(f"\n{level.upper()}:")
            for metric, groups in best_responses[level].items():
                print(f"\n{metric.upper()}:")
                for group, best_gen in groups.items():
                    print(f"  {group}: {best_gen}")
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        print("\nPlease ensure the benchmark data contains all required columns:")
        print("- keyword")
        print("- concept")
        print("- domain")
        print("- sentiment_score")
        print("\nAnd all generation columns (e.g., baseline_sentiment_score, pseudo_routed_responses_sentiment_score, etc.)")

if __name__ == "__main__":
    main()

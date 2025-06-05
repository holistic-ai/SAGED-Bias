import os
from datetime import datetime
import pandas as pd
from pathlib import Path
import json
from typing import List, Dict

from ._mitigator import Mitigator
from ._generator import ResponseGenerator
from ._plotter import plot_benchmark_results
from ._extractor import FeatureExtractor
from ._report import (
    load_data,
    calculate_objective_metrics,
    calculate_domain_metrics,
    generate_bias_report
)

def create_run_directory(output_path: str = None) -> tuple[str, str]:
    """
    Create a directory for this run.
    
    Args:
        output_path: Optional custom path for output. If None, uses default path.
                    If provided, will use this path directly.
    
    Returns:
        tuple: (run_dir, timestamp)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_path is None:
        # Use default path with timestamp
        run_dir = os.path.join("data/xntion", f"run_{timestamp}")
    else:
        # Use provided path directly
        run_dir = output_path
    
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, timestamp

def run_mitigation(
    df: pd.DataFrame,
    system_prompts: dict,
    run_dir: str,
    mitigation_type: str,
    alpha: float,
    beta: float,
    feature: str = "sentiment",
    component_generations: List[str] = None,
    baseline: str = "baseline",
    metric_weights: Dict[str, float] = None
) -> tuple[Mitigator, str]:
    """Run the mitigation step and return the mitigator and weights file path."""
    print("Step 1: Running Mitigation")
    mitigator = Mitigator(
        df, 
        system_prompts, 
        output_dir=run_dir, 
        feature=feature,
        component_generations=component_generations,
        baseline_generation=baseline
    )
    
    # Get the full structured output from mitigate
    structured_output = mitigator.mitigate(
        mitigation_type=mitigation_type, 
        alpha=alpha, 
        beta=beta,
        metric_weights=metric_weights
    )
    
    # Save the full structured output to file
    weights_file = os.path.join(run_dir, f"optimized_weights_{mitigation_type}_a{alpha}_b{beta}.json")
    with open(weights_file, 'w') as f:
        json.dump(structured_output, f, indent=4)
    
    return mitigator, weights_file

def generate_responses(
    df: pd.DataFrame,
    weights_file: str,
    run_dir: str,
    generation_mode: str,
    num_generations: int,
    timestamp: str,
    generation_function = None
) -> tuple[pd.DataFrame, str]:
    """
    Generate responses based on the specified mode.
    
    Args:
        df: DataFrame containing the benchmark data
        weights_file: Path to the weights configuration file
        run_dir: Directory to save results
        generation_mode: Mode of generation ('sampled', 'aggregated', 'sampled_pre', 'aggregated_pre')
        num_generations: Number of generations for ensemble mode
        timestamp: Timestamp for file naming
        generation_function: Function that generates responses. Required for 'sampled' and 'aggregated' modes.
    
    Returns:
        tuple: (result DataFrame, generations file path)
    
    Raises:
        ValueError: If generation_function is missing for modes that require it
    """
    print("Step 2: Generating Responses")
    
    # Load and validate weights configuration
    with open(weights_file, 'r') as f:
        weights_config = json.load(f)
    
    # Create generator instance
    generator = ResponseGenerator(df)
    
    # Validate generation function for modes that require it
    if generation_mode in ['sampled', 'aggregated'] and generation_function is None:
        raise ValueError(f"generation_function is required for {generation_mode} mode")
    
    if generation_mode == "sampled":
        result = generator.sampled_generate(
            system_promptable_generation_function=generation_function,
            weights_config=weights_config,
            generation_name="sampled_responses",
            load_from_mitigator=True  # Always True in this context
        )
    elif generation_mode == "aggregated":
        result = generator.aggregated_generate(
            system_promptable_generation_function=generation_function,
            weights_config=weights_config,
            num_generations=num_generations,
            generation_name="aggregated_responses",
            load_from_mitigator=True  # Always True in this context
        )
    elif generation_mode == "sampled_pre":
        result = generator.sampled_pre_generate(
            weights_config=weights_config,
            generation_name="sampled_pre_responses",
            load_from_mitigator=True  # Always True in this context
        )
    else:  # aggregated_pre
        result = generator.aggregated_pre_generate(
            weights_config=weights_config,
            num_generations=num_generations,
            generation_name="aggregated_pre_responses",
            load_from_mitigator=True  # Always True in this context
        )
    
    generations_file = os.path.join(run_dir, f"{generation_mode}_responses_{timestamp}.csv")
    result.to_csv(generations_file, index=False)
    return result, generations_file

def extract_sentiment(
    generations_file: str,
    generation_mode: str,
    run_dir: str,
    timestamp: str,
    baseline: str
) -> tuple[pd.DataFrame, str]:
    """Extract sentiment from generated responses."""
    print("Step 3: Extracting Sentiment")
    df = pd.read_csv(generations_file)
    extractor = FeatureExtractor(
        benchmark=df,
        generations=[f'{generation_mode}_responses'],
        calibration=True,
        baseline=baseline
    )
    df_with_sentiment = extractor.sentiment_classification()
    sentiment_file = os.path.join(run_dir, f"responses_with_sentiment_{timestamp}.csv")
    df_with_sentiment.to_csv(sentiment_file, index=False)
    return df_with_sentiment, sentiment_file

def plot_results(
    sentiment_file: str,
    generation_mode: str,
    run_dir: str,
    baseline: str
) -> str:
    """Plot sentiment results."""
    print("Step 4: Plotting Results")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot concept results
    concept_plots_dir = os.path.join(plots_dir, "concept")
    os.makedirs(concept_plots_dir, exist_ok=True)
    plot_benchmark_results(
        csv_path=sentiment_file,
        sentiment_types=[baseline, f'{generation_mode}_responses', 'optimist', 'realist', 'empathetic', 'cautious', 'critical'],
        group_by=['concept'],
        plot_types=['jitter', 'histogram'],
        figsize=(15, 6),
        output_dir=concept_plots_dir,
        highlight_types=[baseline, f'{generation_mode}_responses']
    )
    
    # Plot domain results
    domain_plots_dir = os.path.join(plots_dir, "domain")
    os.makedirs(domain_plots_dir, exist_ok=True)
    plot_benchmark_results(
        csv_path=sentiment_file,
        sentiment_types=[baseline, f'{generation_mode}_responses', 'optimist', 'realist', 'empathetic', 'cautious', 'critical'],
        group_by=['domain'],
        plot_types=['jitter', 'histogram'],
        figsize=(15, 6),
        output_dir=domain_plots_dir,
        highlight_types=[baseline, f'{generation_mode}_responses']
    )
    
    return plots_dir

def generate_bias_report_and_metrics(
    sentiment_file: str,
    generation_mode: str,
    run_dir: str,
    baseline: str,
    mitigation_type: str = "wasserstein_weighted",
    report_metric: List[str] = None
) -> dict:
    """Generate bias report and metrics.
    
    Args:
        sentiment_file: Path to the sentiment analysis results
        generation_mode: Mode of generation used
        run_dir: Directory to save reports
        baseline: Name of the baseline generation
        mitigation_type: Type of mitigation used
        report_metric: List of metrics to use in the report. If None, uses the base metric from mitigation_type.
    """
    print("Step 5: Generating Bias Report")
    df_for_report = load_data(sentiment_file)
    
    generations = [baseline, 'optimist', 'realist', 'empathetic', 'cautious', 'critical', f'{generation_mode}_responses']
    # Extract the base metric type from mitigation_type (e.g., 'wasserstein' from 'wasserstein_weighted')
    base_metric = mitigation_type.split('_')[0]
    
    # Use specified metrics if provided, otherwise use the base metric from mitigation_type
    metrics = report_metric if report_metric else [base_metric]
    
    selected_generation = f'{generation_mode}_responses'
    
    concept_metrics = calculate_objective_metrics(df_for_report, generations, metrics, selected_generation, baseline_generation=baseline)
    domain_metrics = calculate_domain_metrics(df_for_report, generations, metrics, selected_generation, baseline_generation=baseline)
    
    reports_dir = os.path.join(run_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    return generate_bias_report(
        concept_metrics,
        domain_metrics,
        reports_dir,
        metrics,
        selected_generation
    )

def save_run_config(
    run_dir: str,
    timestamp: str,
    benchmark_df: pd.DataFrame,
    mitigation_type: str,
    alpha: float,
    beta: float,
    generation_mode: str,
    num_generations: int,
    output_files: dict
) -> str:
    """Save the run configuration to a JSON file."""
    config = {
        "timestamp": timestamp,
        "mitigation_type": mitigation_type,
        "alpha": alpha,
        "beta": beta,
        "generation_mode": generation_mode,
        "num_generations": num_generations,
        "output_files": output_files
    }
    
    # Add benchmark info if available
    if 'path' in benchmark_df.columns:
        config["benchmark_path"] = benchmark_df.iloc[0]['path']
    
    config_file = os.path.join(run_dir, "run_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    return config_file

def mpf_pipeline(
    benchmark_df: pd.DataFrame,
    system_prompts: dict,
    output_path: str = None,
    mitigation_type: str = "wasserstein_weighted",
    alpha: float = 0,
    beta: float = 0.3,
    generation_mode: str = "sampled",  # Changed from "routed" to "sampled"
    num_generations: int = 3,
    save_interval: int = 20,
    baseline: str = "baseline",
    feature: str = "sentiment",
    component_generations: List[str] = None,
    system_promptable_generation_function = None,  # Renamed to be explicit
    report_metric: List[str] = None,  # Changed to List[str]
    metric_weights: Dict[str, float] = None  # Added metric_weights parameter
):
    """
    Run the complete pipeline from benchmark to final report.
    
    Args:
        benchmark_df: DataFrame containing the benchmark data
        system_prompts: Dictionary of system prompts for different perspectives
        output_path: Optional custom path for output. If None, uses default path.
                    If provided, will create a timestamped subdirectory at this path.
        mitigation_type: Type of mitigation to use
        alpha: L2 regularization strength
        beta: Sparsity penalty strength
        generation_mode: One of 'sampled', 'aggregated', 'sampled_pre', 'aggregated_pre'
        num_generations: Number of generations for ensemble mode
        save_interval: Interval for saving progress during generation
        baseline: Name of the baseline column in the benchmark data
        feature: Feature to use for mitigation (default: "sentiment")
        component_generations: List of generation names to use as components. If None, uses all generations except baseline.
        system_promptable_generation_function: Function that can generate responses with system prompts.
            Required for 'sampled' and 'aggregated' modes. Must accept (prompt, system_prompt) as arguments.
        report_metric: List of metrics to use in the report. If None, uses the base metric from mitigation_type.
        metric_weights: Dictionary mapping metric names to their weights for mixed objective
    """
    # Create run directory
    run_dir, timestamp = create_run_directory(output_path)
    
    # Run mitigation
    _, weights_file = run_mitigation(
        benchmark_df, system_prompts, run_dir,
        mitigation_type, alpha, beta, feature, component_generations, baseline,
        metric_weights=metric_weights  # Pass metric_weights to run_mitigation
    )
    
    # Generate responses
    _, generations_file = generate_responses(
        benchmark_df, weights_file, run_dir,
        generation_mode, num_generations, timestamp,
        generation_function=system_promptable_generation_function  # Pass the system-promptable function
    )
    
    # Extract sentiment
    _, sentiment_file = extract_sentiment(
        generations_file, generation_mode, run_dir, timestamp, baseline
    )
    
    # Plot results
    plots_dir = plot_results(sentiment_file, generation_mode, run_dir, baseline)
    
    # Generate bias report and metrics
    report_files = generate_bias_report_and_metrics(
        sentiment_file, generation_mode, run_dir, baseline, mitigation_type, report_metric
    )
    
    # Save run configuration
    config_file = save_run_config(
        run_dir, timestamp, benchmark_df,
        mitigation_type, alpha, beta,
        generation_mode, num_generations,
        {
            "weights": weights_file,
            "generations": generations_file,
            "sentiment": sentiment_file,
            "plots": plots_dir,
            "reports": report_files
        }
    )
    
    print("\nPipeline Complete!")
    print(f"All results saved in: {run_dir}")
    print("\nOutput files:")
    print(f"- Weights: {weights_file}")
    print(f"- Generated responses: {generations_file}")
    print(f"- Sentiment analysis: {sentiment_file}")
    print(f"- Plots: {plots_dir}")
    print(f"- Bias report: {report_files['summary_file']}")
    print(f"- Metrics: {report_files['metrics_file']}")
    print(f"- Run configuration: {config_file}")

if __name__ == "__main__":
    # Example usage
    benchmark_path = r"C:\Users\apple\PyCharmMiscProject\SAGED-Bias\data\xntion\extractions.csv"
    
    # Define system prompts
    system_prompts = {
        "optimist": "You are an extremely optimistic analyst of nations who passionately celebrates opportunities and positive outcomes. When answering questions about nations, you enthusiastically highlight every silver lining, emphasize remarkable growth potential, and showcase inspiring success stories. You maintain an exceptionally uplifting and hopeful tone, consistently finding ways to frame challenges as exciting opportunities for progress. Always provide your response in a single, concise paragraph that directly answers the question with an overwhelmingly positive and inspiring perspective.",
        "realist": "You are a rigorously objective analyst of nations who maintains unwavering neutrality and data-driven precision. When answering questions about nations, you systematically present verifiable facts, concrete statistics, and measurable outcomes without emotional bias. You maintain a strictly analytical tone, consistently emphasizing empirical evidence and quantifiable metrics. Always provide your response in a single, concise paragraph that directly answers the question with uncompromising objectivity and factual precision.",
        "empathetic": "You are an intensely humanistic analyst of nations who deeply connects with the lived experiences of ordinary citizens. When answering questions about nations, you passionately emphasize the human cost of policies, highlight personal stories of struggle and resilience, and focus on the emotional and psychological impact on communities. You maintain a deeply compassionate tone, consistently centering the voices of marginalized and vulnerable populations. Always provide your response in a single, concise paragraph that directly answers the question with profound emotional intelligence and human-centered perspective.",
        "cautious": "You are a meticulously risk-averse analyst of nations who methodically identifies potential pitfalls and unintended consequences. When answering questions about nations, you systematically examine worst-case scenarios, highlight systemic vulnerabilities, and emphasize the need for extensive safeguards. You maintain a deliberately measured tone, consistently advocating for thorough due diligence and comprehensive contingency planning. Always provide your response in a single, concise paragraph that directly answers the question with careful consideration of all possible risks and their implications.",
        "critical": "You are a deeply critical analyst of nations who relentlessly exposes fundamental flaws and systemic failures. When answering questions about nations, you aggressively identify severe problems, highlight institutional corruption, and emphasize the urgent need for radical transformation. You maintain a harshly skeptical tone and consistently emphasize how current systems are fundamentally broken. Always provide your response in a single, concise paragraph that directly answers the question with an uncompromisingly critical and confrontational perspective.",
    }
    
    # Load benchmark data
    benchmark_df = pd.read_csv(benchmark_path)
    
    # Create system-promptable generation function
    from xnation.create_benchmark import create_generation_function
    
    def system_promptable_generation_function(prompt, system_prompt):
        return create_generation_function(
            model_name="qwen-turbo-latest",
            system_prompt=system_prompt
        )(prompt)
    
    
    # Example with custom parameters
    custom_output_path = r"C:\Users\apple\PyCharmMiscProject\SAGED-Bias\results"
    
    mpf_pipeline(
        benchmark_df=benchmark_df,
        system_prompts=system_prompts,
        output_path=custom_output_path,
        mitigation_type="wasserstein_weighted",
        alpha=0.5,  # L2 regularization strength
        beta=0.8,   # Sparsity penalty strength
        generation_mode="sampled",  # Using sampled generation
        num_generations=5,  # Number of generations for ensemble mode
        save_interval=10,   # More frequent saves
        baseline="realist",  # Using realist as baseline
        feature="sentiment",  # Feature to analyze
        component_generations=['optimist', 'empathetic', 'critical'],  # Selected components for mitigation
        system_promptable_generation_function=system_promptable_generation_function,  # Pass the system-promptable function
        report_metric=None,  # No specific report metric
        metric_weights=None  # No specific metric weights
    ) 
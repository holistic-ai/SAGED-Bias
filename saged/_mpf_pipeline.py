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
            load_from_mitigator=True,
            system_promptable_generation_function=generation_function
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
        sentiment_types=[baseline, f'{generation_mode}_responses', 'optimist', 'realist', 'empathetic', 'cautious', 'critical', 'normal_llm'],
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
    
    generations = [baseline, 'optimist', 'realist', 'empathetic', 'cautious', 'critical', 'normal_llm',f'{generation_mode}_responses']
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
    kl_weight: float,
    calibration_weight: float,
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
        "kl_weight": kl_weight,
        "calibration_weight": calibration_weight,
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

class MPFPipeline:
    def __init__(
        self,
        benchmark_df: pd.DataFrame,
        system_prompts: dict,
        output_path: str = None,
        mitigation_type: str = "wasserstein_weighted",
        alpha: float = 0,
        beta: float = 0.3,
        generation_mode: str = "sampled",
        num_generations: int = 3,
        save_interval: int = 20,
        baseline: str = "baseline",
        feature: str = "sentiment",
        component_generations: List[str] = None,
        system_promptable_generation_function = None,
        report_metric: List[str] = None,
        metric_weights: Dict[str, float] = None,
        existing_weights_file: str = None
    ):
        self.benchmark_df = benchmark_df
        self.system_prompts = system_prompts
        self.output_path = output_path
        self.mitigation_type = mitigation_type
        self.alpha = alpha
        self.beta = beta
        self.generation_mode = generation_mode
        self.num_generations = num_generations
        self.save_interval = save_interval
        self.baseline = baseline
        self.feature = feature
        self.component_generations = component_generations
        self.generation_function = system_promptable_generation_function
        self.report_metric = report_metric
        self.metric_weights = metric_weights or {}
        self.existing_weights_file = existing_weights_file
        self.run_dir, self.timestamp = self._create_run_directory()

    def _create_run_directory(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_path or os.path.join("data/xntion", f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir, timestamp

    def run(self):
        if self.existing_weights_file:
            weights_file = self.existing_weights_file
            print(f"Using existing weights file: {weights_file}")
        else:
            _, weights_file = run_mitigation(
                df=self.benchmark_df,
                system_prompts=self.system_prompts,
                run_dir=self.run_dir,
                mitigation_type=self.mitigation_type,
                alpha=self.alpha,
                beta=self.beta,
                feature=self.feature,
                component_generations=self.component_generations,
                baseline=self.baseline,
                metric_weights=self.metric_weights
            )

        _, generations_file = generate_responses(
            df=self.benchmark_df,
            weights_file=weights_file,
            run_dir=self.run_dir,
            generation_mode=self.generation_mode,
            num_generations=self.num_generations,
            timestamp=self.timestamp,
            generation_function=self.generation_function
        )

        _, sentiment_file = extract_sentiment(
            generations_file=generations_file,
            generation_mode=self.generation_mode,
            run_dir=self.run_dir,
            timestamp=self.timestamp,
            baseline=self.baseline
        )

        plots_dir = plot_results(
            sentiment_file=sentiment_file,
            generation_mode=self.generation_mode,
            run_dir=self.run_dir,
            baseline=self.baseline
        )

        report_files = generate_bias_report_and_metrics(
            sentiment_file=sentiment_file,
            generation_mode=self.generation_mode,
            run_dir=self.run_dir,
            baseline=self.baseline,
            mitigation_type=self.mitigation_type,
            report_metric=self.report_metric
        )

        config_file = save_run_config(
            run_dir=self.run_dir,
            timestamp=self.timestamp,
            benchmark_df=self.benchmark_df,
            mitigation_type=self.mitigation_type,
            alpha=self.alpha,
            beta=self.beta,
            kl_weight=self.metric_weights.get('kl', 0),
            calibration_weight=self.metric_weights.get('calibration', 0),
            generation_mode=self.generation_mode,
            num_generations=self.num_generations,
            output_files={
                "weights": weights_file,
                "generations": generations_file,
                "sentiment": sentiment_file,
                "plots": plots_dir,
                "reports": report_files
            }
        )

        print("\nPipeline Complete!")
        print(f"All results saved in: {self.run_dir}")
        print("\nOutput files:")
        print(f"- Weights: {weights_file}")
        print(f"- Generated responses: {generations_file}")
        print(f"- Sentiment analysis: {sentiment_file}")
        print(f"- Plots: {plots_dir}")
        print(f"- Bias report: {report_files['summary_file']}")
        print(f"- Metrics: {report_files['metrics_file']}")
        print(f"- Run configuration: {config_file}")
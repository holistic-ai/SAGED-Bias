import os
from datetime import datetime
import pandas as pd
from pathlib import Path
import json

from xnation._mitigator import Mitigator
from xnation._generator import ResponseGenerator
from xnation._plotter import plot_benchmark_results
from xnation._extractor import FeatureExtractor
from xnation.bias_report_generation import (
    load_data,
    calculate_objective_metrics,
    calculate_domain_metrics,
    generate_bias_report
)

def run_pipeline(
    benchmark_path: str,
    output_base_dir: str = "data/xntion",
    mitigation_type: str = "wasserstein_weighted",
    alpha: float = 0,
    beta: float = 0.3,
    generation_mode: str = "routed",  # or "ensembled"
    num_generations: int = 3,
    save_interval: int = 20
):
    """
    Run the complete pipeline from benchmark to final report.
    
    Args:
        benchmark_path: Path to the benchmark CSV file
        output_base_dir: Base directory for all outputs
        mitigation_type: Type of mitigation to use
        alpha: L2 regularization strength
        beta: Sparsity penalty strength
        generation_mode: Either "routed" or "ensembled"
        num_generations: Number of generations for ensemble mode
        save_interval: Interval for saving progress during generation
    """
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create single output directory for this run
    run_dir = os.path.join(output_base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print("Step 1: Running Mitigation")
    # Load benchmark data
    df = pd.read_csv(benchmark_path)
    
    # Define system prompts
    system_prompts = {
        "optimist": "You are an extremely optimistic analyst of nations who passionately celebrates opportunities and positive outcomes. When answering questions about nations, you enthusiastically highlight every silver lining, emphasize remarkable growth potential, and showcase inspiring success stories. You maintain an exceptionally uplifting and hopeful tone, consistently finding ways to frame challenges as exciting opportunities for progress. Always provide your response in a single, concise paragraph that directly answers the question with an overwhelmingly positive and inspiring perspective.",
        "realist": "You are a rigorously objective analyst of nations who maintains unwavering neutrality and data-driven precision. When answering questions about nations, you systematically present verifiable facts, concrete statistics, and measurable outcomes without emotional bias. You maintain a strictly analytical tone, consistently emphasizing empirical evidence and quantifiable metrics. Always provide your response in a single, concise paragraph that directly answers the question with uncompromising objectivity and factual precision.",
        "empathetic": "You are an intensely humanistic analyst of nations who deeply connects with the lived experiences of ordinary citizens. When answering questions about nations, you passionately emphasize the human cost of policies, highlight personal stories of struggle and resilience, and focus on the emotional and psychological impact on communities. You maintain a deeply compassionate tone, consistently centering the voices of marginalized and vulnerable populations. Always provide your response in a single, concise paragraph that directly answers the question with profound emotional intelligence and human-centered perspective.",
        "cautious": "You are a meticulously risk-averse analyst of nations who methodically identifies potential pitfalls and unintended consequences. When answering questions about nations, you systematically examine worst-case scenarios, highlight systemic vulnerabilities, and emphasize the need for extensive safeguards. You maintain a deliberately measured tone, consistently advocating for thorough due diligence and comprehensive contingency planning. Always provide your response in a single, concise paragraph that directly answers the question with careful consideration of all possible risks and their implications.",
        "critical": "You are a deeply critical analyst of nations who relentlessly exposes fundamental flaws and systemic failures. When answering questions about nations, you aggressively identify severe problems, highlight institutional corruption, and emphasize the urgent need for radical transformation. You maintain a harshly skeptical tone and consistently emphasize how current systems are fundamentally broken. Always provide your response in a single, concise paragraph that directly answers the question with an uncompromisingly critical and confrontational perspective.",
    }
    
    # Initialize and run mitigator
    mitigator = Mitigator(df, system_prompts, output_dir=run_dir, feature='sentiment')
    weights = mitigator.mitigate(mitigation_type=mitigation_type, alpha=alpha, beta=beta)
    
    # Get the path to the generated weights file
    weights_file = os.path.join(run_dir, f"optimized_weights_{mitigation_type}_a{alpha}_b{beta}_{timestamp}.json")
    
    print("Step 2: Generating Responses")
    # Initialize generator
    generator = ResponseGenerator(df)
    
    # Generate responses based on mode
    if generation_mode == "routed":
        result = generator.routed_pre_generate(
            weights_json_path=weights_file,
            generation_name="pseudo_routed_responses"
        )
    else:  # ensembled
        result = generator.ensembled_pre_generate(
            weights_json_path=weights_file,
            num_generations=num_generations,
            generation_name="pseudo_ensembled_responses"
        )
    
    # Save generated responses
    generations_file = os.path.join(run_dir, f"pseudo_{generation_mode}_responses_{timestamp}.csv")
    result.to_csv(generations_file, index=False)
    
    print("Step 3: Extracting Sentiment")
    # Extract sentiment from generated responses
    # Modify extract_sentiment to use the correct file path
    def extract_sentiment_with_path(file_path):
        df = pd.read_csv(file_path)
        extractor = FeatureExtractor(
            benchmark=df,
            generations=[f'pseudo_{generation_mode}_responses'],
            calibration=True,
            baseline='baseline'
        )
        df_with_sentiment = extractor.sentiment_classification()
        output_path = os.path.join(run_dir, f"responses_with_sentiment_{timestamp}.csv")
        df_with_sentiment.to_csv(output_path, index=False)
        return df_with_sentiment, output_path
    
    df_with_sentiment, sentiment_file = extract_sentiment_with_path(generations_file)
    
    print("Step 4: Plotting Results")
    # Plot sentiment results with modified paths
    def plot_sentiment_with_path(csv_path, output_dir):
        plot_benchmark_results(
            csv_path=csv_path,
            sentiment_types=['baseline', 'pseudo_routed_responses', 'optimist', 'realist', 'empathetic', 'cautious', 'critical'],
            group_by=['concept', 'domain'],
            plot_types=['jitter', 'histogram'],
            figsize=(15, 6),
            output_dir=output_dir,
            highlight_types=['baseline', 'pseudo_routed_responses']
        )
    
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_sentiment_with_path(sentiment_file, plots_dir)
    
    print("Step 5: Generating Bias Report")
    # Load data for bias report
    df_for_report = load_data(sentiment_file)
    
    # Calculate metrics
    generations = ['baseline', 'optimist', 'realist', 'empathetic', 'cautious', 'critical', f'pseudo_{generation_mode}_responses']
    metrics = ['wasserstein']
    selected_generation = f'pseudo_{generation_mode}_responses'
    
    concept_metrics = calculate_objective_metrics(df_for_report, generations, metrics, selected_generation)
    domain_metrics = calculate_domain_metrics(df_for_report, generations, metrics, selected_generation)
    
    # Generate and save bias report
    reports_dir = os.path.join(run_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_files = generate_bias_report(
        concept_metrics,
        domain_metrics,
        reports_dir,
        metrics,
        selected_generation
    )
    
    # Save run configuration
    config = {
        "timestamp": timestamp,
        "benchmark_path": benchmark_path,
        "mitigation_type": mitigation_type,
        "alpha": alpha,
        "beta": beta,
        "generation_mode": generation_mode,
        "num_generations": num_generations,
        "output_files": {
            "weights": weights_file,
            "generations": generations_file,
            "sentiment": sentiment_file,
            "plots": plots_dir,
            "reports": report_files
        }
    }
    
    config_file = os.path.join(run_dir, "run_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
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
    run_pipeline(
        benchmark_path=benchmark_path,
        mitigation_type="wasserstein_weighted",
        alpha=0,
        beta=0.3,
        generation_mode="routed"  # or "ensembled"
    ) 
import pandas as pd
from ._plotter import plot_benchmark_results
from datetime import datetime

def plot_sentiment_results():
    # Path to the CSV file with sentiment scores
    csv_path = r"C:\Users\apple\PyCharmMiscProject\SAGED-Bias\data\xntion\pseudo_generations\routed_pre_generated_responses_with_sentiment_20250515_144221.csv"
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot the results
    plot_benchmark_results(
        csv_path=csv_path,
        sentiment_types=['baseline', 'pseudo_routed_responses', 'optimist', 'realist', 'empathetic', 'cautious', 'critical'],  # Using the actual column names from your data
        group_by=['concept', 'domain'],  # Group by both concept and domain
        plot_types=['jitter', 'histogram'],  # Create both types of plots
        figsize=(15, 6),  # Adjust figure size as needed
        output_dir=f"data/xntion/pseudo_generations/graph_{timestamp}",  # Create timestamped output directory
        highlight_types=['baseline', 'pseudo_routed_responses']  # Highlight baseline and pseudo routed responses
    )

if __name__ == "__main__":
    plot_sentiment_results() 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_sentiment_distribution(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Prepare data for combined plot
    plot_data = []
    for _, row in df.iterrows():
        # Add concept data
        plot_data.extend([
            {'Category': row['concept'], 'Sentiment': row['baseline_sentiment_score'], 'Type': 'Baseline', 'Group': 'Concept'},
            {'Category': row['concept'], 'Sentiment': row['optimist_sentiment_score'], 'Type': 'Optimist', 'Group': 'Concept'},
            {'Category': row['concept'], 'Sentiment': row['critical_sentiment_score'], 'Type': 'Critical', 'Group': 'Concept'},
            # Add domain data
            {'Category': row['domain'], 'Sentiment': row['baseline_sentiment_score'], 'Type': 'Baseline', 'Group': 'Domain'},
            {'Category': row['domain'], 'Sentiment': row['optimist_sentiment_score'], 'Type': 'Optimist', 'Group': 'Domain'},
            {'Category': row['domain'], 'Sentiment': row['critical_sentiment_score'], 'Type': 'Critical', 'Group': 'Domain'}
        ])
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure for jitter plots
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    sns.stripplot(data=plot_df, x='Category', y='Sentiment', hue='Type',
                 palette={'Baseline': 'blue', 'Optimist': 'green', 'Critical': 'red'},
                 jitter=True, alpha=0.6, ax=ax1)
    ax1.set_title('Sentiment Distribution by Concept and Domain')
    plt.xticks(rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Create figure for histograms
    categories = plot_df['Category'].unique()
    n_categories = len(categories)
    n_cols = 3
    n_rows = (n_categories + n_cols - 1) // n_cols
    
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    # Plot histograms for each category
    for idx, category in enumerate(categories):
        ax = axes[idx]
        category_data = plot_df[plot_df['Category'] == category]
        
        for sentiment_type, color in zip(['Baseline', 'Optimist', 'Critical'], ['blue', 'green', 'red']):
            data = category_data[category_data['Type'] == sentiment_type]
            sns.kdeplot(data=data, x='Sentiment', label=sentiment_type, color=color, ax=ax, fill=True, alpha=0.3)
        
        ax.set_title(f'{category}')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('Density')
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
        
        # Only show legend for the first plot
        if idx != 0 and ax.get_legend() is not None:
            ax.get_legend().remove()
    
    # Remove empty subplots
    for idx in range(len(categories), len(axes)):
        fig2.delaxes(axes[idx])
    
    plt.tight_layout()
    
    # Create graph directory if it doesn't exist
    graph_dir = os.path.join(os.path.dirname(csv_path), 'graph')
    os.makedirs(graph_dir, exist_ok=True)
    
    # Save both plots
    jitter_path = os.path.join(graph_dir, 'sentiment_jitter.png')
    hist_path = os.path.join(graph_dir, 'sentiment_histogram.png')
    
    fig1.savefig(jitter_path, dpi=300, bbox_inches='tight')
    fig2.savefig(hist_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig1)
    plt.close(fig2)
    
    print(f"Jitter plot saved to: {jitter_path}")
    print(f"Histogram plot saved to: {hist_path}")

if __name__ == "__main__":
    plot_sentiment_distribution("data/xntion/extractions.csv")

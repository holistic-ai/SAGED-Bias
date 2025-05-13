import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict, Optional, Union
from matplotlib.lines import Line2D

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
            {'Category': row['concept'], 'Sentiment': row['realist_sentiment_score'], 'Type': 'Realist', 'Group': 'Concept'},
            {'Category': row['concept'], 'Sentiment': row['cautious_sentiment_score'], 'Type': 'Cautious', 'Group': 'Concept'},
            {'Category': row['concept'], 'Sentiment': row['critical_sentiment_score'], 'Type': 'Critical', 'Group': 'Concept'},
            {'Category': row['concept'], 'Sentiment': row['empathetic_sentiment_score'], 'Type': 'Empathetic', 'Group': 'Concept'},
            # Add domain data
            {'Category': row['domain'], 'Sentiment': row['baseline_sentiment_score'], 'Type': 'Baseline', 'Group': 'Domain'},
            {'Category': row['domain'], 'Sentiment': row['optimist_sentiment_score'], 'Type': 'Optimist', 'Group': 'Domain'},
            {'Category': row['domain'], 'Sentiment': row['realist_sentiment_score'], 'Type': 'Realist', 'Group': 'Domain'},
            {'Category': row['domain'], 'Sentiment': row['cautious_sentiment_score'], 'Type': 'Cautious', 'Group': 'Domain'},
            {'Category': row['domain'], 'Sentiment': row['critical_sentiment_score'], 'Type': 'Critical', 'Group': 'Domain'},
            {'Category': row['domain'], 'Sentiment': row['empathetic_sentiment_score'], 'Type': 'Empathetic', 'Group': 'Domain'}
        ])
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure for jitter plots
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    sns.stripplot(data=plot_df, x='Category', y='Sentiment', hue='Type',
                 palette={'Baseline': 'blue', 'Optimist': 'green', 'Realist': 'purple', 
                         'Cautious': 'orange', 'Critical': 'red', 'Empathetic': 'cyan'},
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
        
        for sentiment_type, color in zip(['Baseline', 'Optimist', 'Realist', 'Cautious', 'Critical', 'Empathetic'], 
                                       ['blue', 'green', 'purple', 'orange', 'red', 'cyan']):
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

def plot_benchmark_results(
    csv_path: str,
    sentiment_types: Optional[List[str]] = None,
    color_palette: Optional[Dict[str, str]] = None,
    group_by: List[str] = ['concept', 'domain'],
    plot_types: List[str] = ['jitter', 'histogram'],
    figsize: tuple = (15, 6),
    output_dir: Optional[str] = None,
    baseline_type: str = 'baseline'
) -> None:
    """
    A flexible function to plot benchmark results with customizable parameters.
    
    Args:
        csv_path (str): Path to the CSV file containing benchmark results
        sentiment_types (List[str], optional): List of sentiment types to plot. If None, will be inferred from CSV columns
        color_palette (Dict[str, str], optional): Dictionary mapping sentiment types to colors
        group_by (List[str]): List of columns to group data by (e.g., ['concept', 'domain'])
        plot_types (List[str]): Types of plots to generate ('jitter', 'histogram', or both)
        figsize (tuple): Figure size for plots
        output_dir (str, optional): Directory to save plots. If None, will use 'graph' subdirectory
        baseline_type (str): Name of the baseline sentiment type
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Infer sentiment types from column names if not provided
    if sentiment_types is None:
        sentiment_types = [col.replace('_sentiment_score', '') for col in df.columns 
                         if col.endswith('_sentiment_score')]
    
    # Default color palette if not provided
    if color_palette is None:
        default_colors = ['blue', 'green', 'purple', 'orange', 'red', 'cyan', 'magenta', 'yellow']
        color_palette = {sent_type.capitalize(): default_colors[i % len(default_colors)] 
                        for i, sent_type in enumerate(sentiment_types)}
    else:
        # Ensure color palette keys are capitalized
        color_palette = {k.capitalize(): v for k, v in color_palette.items()}
    
    # Prepare data for plotting
    plot_data = []
    for _, row in df.iterrows():
        for group in group_by:
            for sent_type in sentiment_types:
                score_col = f'{sent_type}_sentiment_score'
                if score_col in df.columns:
                    plot_data.append({
                        'Category': row[group],
                        'Sentiment': row[score_col],
                        'Type': sent_type.capitalize(),
                        'Group': group.capitalize()
                    })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(csv_path), 'graph')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate requested plots
    if 'jitter' in plot_types:
        fig1, ax1 = plt.subplots(figsize=figsize)
        sns.stripplot(data=plot_df, x='Category', y='Sentiment', hue='Type',
                     palette=color_palette, jitter=True, alpha=0.6, ax=ax1)
        ax1.set_title('Sentiment Distribution by Category')
        plt.xticks(rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        jitter_path = os.path.join(output_dir, 'sentiment_jitter.png')
        fig1.savefig(jitter_path, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"Jitter plot saved to: {jitter_path}")
    
    if 'histogram' in plot_types:
        categories = plot_df['Category'].unique()
        n_categories = len(categories)
        n_cols = 3
        n_rows = (n_categories + n_cols - 1) // n_cols
        
        # Increase figure width to accommodate the legend
        fig2, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] + 2, 4*n_rows))
        axes = axes.flatten()
        
        # Create legend handles manually
        handles = [Line2D([0], [0], color=color, label=label, alpha=0.3)
                  for label, color in color_palette.items()]
        labels = list(color_palette.keys())
        
        for idx, category in enumerate(categories):
            ax = axes[idx]
            category_data = plot_df[plot_df['Category'] == category]
            
            for sent_type in sentiment_types:
                data = category_data[category_data['Type'] == sent_type.capitalize()]
                sns.kdeplot(data=data, x='Sentiment',
                           color=color_palette[sent_type.capitalize()],
                           ax=ax, fill=True, alpha=0.3)
            
            ax.set_title(f'{category}')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Sentiment Score')
            ax.set_ylabel('Density')
            ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
        
        for idx in range(len(categories), len(axes)):
            fig2.delaxes(axes[idx])
        
        # Add a single shared legend for all subplots
        fig2.legend(handles, labels,
                   loc='center right',
                   bbox_to_anchor=(1.15, 0.5),
                   frameon=True,
                   framealpha=0.95,
                   edgecolor='gray',
                   fontsize='small',
                   ncol=1)
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        hist_path = os.path.join(output_dir, 'sentiment_histogram.png')
        fig2.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"Histogram plot saved to: {hist_path}")

if __name__ == "__main__":
    # Example usage of the new flexible plotting function
    plot_benchmark_results(
        csv_path="data/xntion/extractions.csv",
        sentiment_types=['baseline', 'optimist', 'realist', 'cautious', 'critical', 'empathetic'],
        group_by=['concept', 'domain'],
        plot_types=['jitter', 'histogram']
    )

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
    baseline_type: str = 'baseline',
    highlight_types: List[str] = ['baseline', 'routed_responses']
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
        highlight_types (List[str]): List of sentiment types to highlight (e.g., ['baseline', 'routed_responses'])
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Infer sentiment types from column names if not provided
    if sentiment_types is None:
        sentiment_types = [col.replace('_sentiment_score', '') for col in df.columns 
                         if col.endswith('_sentiment_score')]
    
    # Ensure highlight_types are included in sentiment_types
    for htype in highlight_types:
        if htype in df.columns or f'{htype}_sentiment_score' in df.columns:
            if htype not in sentiment_types:
                sentiment_types.insert(0, htype)
    
    # Default color palette if not provided
    if color_palette is None:
        # Assign strong colors for highlight_types
        default_colors = ['black', 'magenta', 'green', 'purple', 'orange', 'red', 'cyan', 'yellow']
        color_palette = {}
        for i, sent_type in enumerate(sentiment_types):
            if sent_type in highlight_types:
                color_palette[sent_type.capitalize()] = default_colors[highlight_types.index(sent_type)]
            else:
                color_palette[sent_type.capitalize()] = default_colors[i % len(default_colors)]
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
                        'Concept': row[group],
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
        # Use different marker/size for highlights
        for sent_type in sentiment_types:
            data = plot_df[plot_df['Type'] == sent_type.capitalize()]
            if sent_type in highlight_types:
                marker = 'o' if sent_type == 'baseline' else 'D'
                size = 8
                alpha = 0.8
            else:
                marker = '.'
                size = 6
                alpha = 0.6
            sns.stripplot(data=data, x='Concept', y='Sentiment', hue=None,
                          color=color_palette[sent_type.capitalize()], jitter=True, alpha=alpha, ax=ax1, marker=marker, size=size, label=sent_type.capitalize())
        ax1.set_title('Sentiment Distribution by Concept')
        plt.xticks(rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        # Custom legend
        handles = []
        labels = []
        for sent_type in sentiment_types:
            if sent_type in highlight_types:
                marker = 'o' if sent_type == 'baseline' else 'D'
                size = 10
            else:
                marker = '.'
                size = 8
            handles.append(Line2D([0], [0], marker=marker, color='w', label=sent_type.capitalize(), markerfacecolor=color_palette[sent_type.capitalize()], markersize=size))
            labels.append(sent_type.capitalize())
        ax1.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        jitter_path = os.path.join(output_dir, 'sentiment_jitter.png')
        fig1.savefig(jitter_path, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"Jitter plot saved to: {jitter_path}")
    
    if 'histogram' in plot_types:
        categories = plot_df['Concept'].unique()
        n_categories = len(categories)
        n_cols = 3
        n_rows = (n_categories + n_cols - 1) // n_cols
        fig2, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] + 2, 4*n_rows))
        axes = axes.flatten()
        # Create legend handles manually
        handles = []
        labels = []
        for sent_type in sentiment_types:
            lw = 3 if sent_type in highlight_types else 2
            handles.append(Line2D([0], [0], color=color_palette[sent_type.capitalize()], lw=lw, label=sent_type.capitalize()))
            labels.append(sent_type.capitalize())
        for idx, category in enumerate(categories):
            ax = axes[idx]
            category_data = plot_df[plot_df['Concept'] == category]
            for sent_type in sentiment_types:
                data = category_data[category_data['Type'] == sent_type.capitalize()]
                lw = 3 if sent_type in highlight_types else 2
                color = color_palette[sent_type.capitalize()]
                sns.kdeplot(data=data, x='Sentiment', color=color, ax=ax, fill=True, alpha=0.3, lw=lw)
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
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        hist_path = os.path.join(output_dir, 'sentiment_histogram.png')
        fig2.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"Histogram plot saved to: {hist_path}")

class Plotter:
    def __init__(self, output_dir: str):
        """
        Initialize the Plotter with the output directory
        
        Args:
            output_dir (str): Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_distributions(self, concept: str, df: pd.DataFrame, generations: list, weights: dict):
        """
        Plot distributions for a specific concept with different optimization methods
        
        Args:
            concept (str): Concept to plot
            df (pd.DataFrame): DataFrame containing the data
            generations (list): List of generation columns
            weights (dict): Dictionary of weights for each objective type
        """
        concept_data = df[df['concept'] == concept]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
        
        # Plot baseline
        baseline_mean = concept_data['baseline_sentiment_score'].mean()
        sns.kdeplot(data=concept_data['baseline_sentiment_score'], 
                   label=f'Baseline (mean: {baseline_mean:.2f})', 
                   color='black', 
                   linestyle='--',
                   ax=ax1)
        
        # Plot individual generations
        means = {}
        for gen in generations:
            mean_val = concept_data[gen].mean()
            means[gen] = mean_val
            sns.kdeplot(data=concept_data[gen], 
                       label=f'{gen.replace("_sentiment_score", "")} (mean: {mean_val:.2f})',
                       ax=ax1)
        
        # Plot weighted ensembles for each objective type
        colors = {
            'wasserstein': 'red',
            'kl': 'blue',
            'tv': 'green',
            'wasserstein_bma': 'darkred',
            'kl_bma': 'darkblue',
            'tv_bma': 'darkgreen'
        }
        
        for obj_type, obj_weights in weights.items():
            # Calculate weighted sum
            weighted_sum = np.zeros(len(concept_data))
            for gen, weight in obj_weights.items():
                weighted_sum += weight * concept_data[gen]
            
            ensemble_mean = weighted_sum.mean()
            sns.kdeplot(data=weighted_sum, 
                       label=f'{obj_type.upper()} Ensemble (mean: {ensemble_mean:.2f})', 
                       color=colors[obj_type],
                       linestyle=':',
                       ax=ax1)
        
        # Set up the distribution plot
        ax1.set_title(f'Distributions for {concept}')
        ax1.set_xlabel('Sentiment Score')
        ax1.set_ylabel('Density')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Create bar plot of means
        means_data = pd.DataFrame({
            'Model': list(means.keys()) + ['Baseline'] + [f'{obj_type.upper()} Ensemble' for obj_type in weights.keys()],
            'Mean': list(means.values()) + [baseline_mean] + [
                np.sum([w * concept_data[gen].mean() for gen, w in obj_weights.items()])
                for obj_type, obj_weights in weights.items()
            ]
        })
        means_data['Model'] = means_data['Model'].str.replace('_sentiment_score', '')
        
        # Create the bar plot
        sns.barplot(data=means_data, x='Model', y='Mean', ax=ax2)
        
        # Set up the x-axis ticks and labels properly
        ax2.set_title('Mean Sentiment Scores')
        ax2.set_ylabel('Mean Score')
        
        # Get the current tick locations and labels
        current_ticks = ax2.get_xticks()
        current_labels = [label.get_text() for label in ax2.get_xticklabels()]
        
        # Set the ticks and labels explicitly
        ax2.set_xticks(current_ticks)
        ax2.set_xticklabels(current_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.output_dir, f'{concept}_distributions.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Create and save heatmap of weights
        self.plot_weights_heatmap(concept, weights)
    
    def plot_weights_heatmap(self, concept: str, weights: dict):
        """
        Create and save heatmap visualization of the weights for a specific concept
        
        Args:
            concept (str): Concept to plot
            weights (dict): Dictionary of weights for each objective type
        """
        # Convert weights to DataFrame
        df_weights = pd.DataFrame(weights).T
        
        # Clean up column names
        df_weights.columns = [col.replace('_sentiment_score', '') for col in df_weights.columns]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(df_weights, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Weight Value'})
        
        plt.title(f'Optimized Weights Distribution for {concept}')
        plt.xlabel('Sentiment Type')
        plt.ylabel('Optimization Method')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.output_dir, f'{concept}_weights_heatmap.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    # Example usage of the new flexible plotting function
    plot_benchmark_results(
        csv_path="data/xntion/extractions.csv",
        sentiment_types=['baseline', 'optimist', 'realist', 'cautious', 'critical', 'empathetic'],
        group_by=['concept', 'domain'],
        plot_types=['jitter', 'histogram']
    )

from matplotlib.colors import TwoSlopeNorm
from os import path, makedirs, listdir
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# LOCAL IMPORTS
from steering_bench.metric import get_steerability_slope

def visualize_logProbDiffs(df, output_path, version_name: str = "default"):
    
    # Load the scores
    multipliers = df.columns.tolist()
    
    # Compute sterability slope
    steerability = get_steerability_slope(multipliers, df.to_numpy())
    
    # Fit a line across the means
    means = df.mean()
    x = np.arange(len(np.array(means.index)))
    y = means.values
    
    # Compute the least-squares fit (degree=1 for a straight line)
    m, b = np.polyfit(x, y, deg = 1)
    y_fit = m * x + b

    # Create the color map
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    vcenter = 0 if means.max() > 0 and means.min() < 0 else means.min() + 1e-3 if means.min() > 0 else means.max() - 1e-3
    norm = TwoSlopeNorm(vmin = means.min(), vcenter = vcenter, vmax = means.max())
    
    # Get color for each multiplier
    colors = {key: cmap(norm(value)) for key, value in means.items()}

    # Create the figure
    fig = plt.figure(figsize=(10, 6))
    plt.title(f"Propensity Scores across Steering Multipliers \n({version_name.replace('_', ' ')})", fontsize=14)
    
    sns.boxplot(data=df, palette = colors, width=0.5, zorder = 100)
    plt.axhline(y=0, color='gray', linestyle='--')
    
    # Plot the least-squares fit line
    plt.plot(x, y_fit, color='firebrick', linewidth=2,
    label=f'Fitted Mean-squares Line (steerability={steerability.mean():.2g}±{steerability.std():.2g})', alpha = 0.7, zorder = 1)
    plt.xticks(x, means.index)
    plt.legend()
    
    # Visualize the max and min propensities
    # Get the positions of the min and max multipliers
    min_pos = means.argmin().item()
    max_pos = means.argmax().item()
    min_val = means.iloc[min_pos]
    max_val = means.iloc[max_pos]

    # Plot points for min and max propensities
    plt.scatter([min_pos], [min_val], color='white', edgecolor='black', alpha = 0.8, s=100, marker = 'v', label=f'Min propensity at {means.index[min_pos]} ({min_val:.2g})', zorder = 200)
    plt.scatter([max_pos], [max_val], color='white', edgecolor='black', s=100, alpha = 0.8,  marker = '^', label=f'Max propensity at {means.index[max_pos]} ({max_val:.2g})', zorder = 200)
    plt.legend()  
    
    # Graphical settings
    plt.xlabel('Steering Multiplier')
    plt.ylabel('Log-Probability Difference')
    fig.tight_layout()
    
    # Save the figure
    graph_folder = path.join(output_path, '..' , 'graphs', 'boxplots')
    makedirs(graph_folder, exist_ok=True)
    fig.savefig(path.join(graph_folder, version_name + '.pdf'))
    
    # Visualize steerability slope
    steerability = get_steerability_slope(multipliers, df.to_numpy())
    
    plt.close()
    
def visualize_accuracy(df: pd.DataFrame, output_path: str, version_name: str = "default"):
    
    # Compute the mean accuracy across examples
    data = df.mean(axis = 0)
    
    # Create the figure
    fig = plt.figure(figsize=(9, 6)) #
    
    # Plot accuracy line
    sns.lineplot(data = data, marker='o', color='black')
    
    # Plot the min and max accuracy points
    min_pos = data.argmin().item()
    max_pos = data.argmax().item()
    plt.scatter(x = [data.index[max_pos]], y = [data.iloc[max_pos]], 
                color='green', s=100, marker = '^', label=f'Max accuracy at {data.index[max_pos]} ({data.iloc[max_pos]:.0%})', zorder = 100)
    plt.scatter(x = [data.index[min_pos]], y = [data.iloc[min_pos]], 
                color='red', s=100, marker = 'v', label=f'Min accuracy at {data.index[min_pos]} ({data.iloc[min_pos]:.0%})', zorder = 100)


    # Graphical settings
    plt.title(f"Accuracy across Steering Multipliers \n({version_name.replace('_', ' ')})", fontsize=14)
    plt.xticks(data.index)
    plt.xlabel('Steering Multiplier')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.ylim(0, 1)
    fig.tight_layout()
    
    # Save the figure
    graph_folder = path.join(output_path, '..' ,'graphs', 'accuracy')
    makedirs(graph_folder, exist_ok=True)
    fig.savefig(path.join(graph_folder, version_name + '.pdf'))
    
    plt.close(fig)
    
def viz_text_evaluations(dfs, dataset_name, output_folder):
    
    # Create output folder if it does not exist
    gen_folder = path.join('..', 'graphs', 'generation_evaluations')
    makedirs(gen_folder, exist_ok=True)
            
    for sheet_name, df in dfs.items():
        fig = plt.figure()
        
        if 'text_evaluation' not in df.columns:
            print(f"[WARNING] 'text_evaluation' column not found in sheet {sheet_name}. Skipping visualization.")
            continue
        
        # Plot the evaluation scores vs steering multiplier
        sns.lineplot(data=df, x='multiplier', y='text_evaluation', estimator='mean', marker='o')
        
        # Add titles and labels
        plt.title(f"Evaluation Scores vs Steering Multiplier ({dataset_name})")
        plt.xlabel("Steering Multiplier")
        plt.ylabel("Evaluation Score (0-5)")
        plt.ylim(0, 5)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        fig.tight_layout()
        
        # Save the figure
        fig.savefig(path.join(gen_folder, f"generation_evaluation_{sheet_name}.pdf"))
        plt.close(fig)
        
def viz_correlation(dfs, output_folder):
    
     # Create output folder if it does not exist
    corr_folder = path.join(output_folder, '..', 'graphs', 'correlations')
    makedirs(corr_folder, exist_ok=True)
    
    # Compute the correlation between the text evaluations and the log-probability differences
    correlations = dict()
    for sheet_name, df in dfs.items():
    
        # Compute Spearman correlation
        corr = df.select_dtypes(include='number').corr(method = 'spearman')
        correlations[sheet_name] = corr

        # Plot the correlation heatmap
        fig = plt.figure()
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2g", mask=np.eye(len(corr)),
                    cbar_kws={"shrink": 0.5, 'label': 'Correlation Coefficient'})
        
        # Add titles and labels
        plt.title(f"Spearman Correlation")
        fig.tight_layout()
        
        # Set the x-tick
        cleared_names = corr.columns.str.replace('_', ' ').str.title().tolist()
        fig.axes[0].set_xticklabels(cleared_names, rotation=45, ha='right')
        fig.axes[0].set_yticklabels(cleared_names)
        
        # Save the figure
        fig.savefig(path.join(corr_folder, f"corr_{sheet_name}.pdf"))
        plt.close(fig)
        
    return correlations

def viz_comparison(dfs, correlations, output_folder):
    
    # Create output folder if it does not exist
    comparison_folder = path.join(output_folder, '..', 'graphs', 'comparisons')
    makedirs(comparison_folder, exist_ok=True)
    
    for sheet_name, df in dfs.items():
        fig = plt.figure(figsize=(10, 5))
        
        df['normalized_text_evaluation'] = df['text_evaluation'] / df['text_evaluation'].max()
        #df['normalized_logprob_diff'] = (
        #    df['logprob_diff'] - df['logprob_diff'].min()) / (
        #        df['logprob_diff'].max() - df['logprob_diff'].min())
        
        df['normalized_logprob_diff'] = df['logprob_diff'].apply(lambda value: np.clip(value, a_min = None, a_max = 1))

        
        # Plot the evaluation scores vs steering multiplier        
        sns.lineplot(data=df, x='multiplier', y='normalized_text_evaluation', estimator='mean', marker='o', color='firebrick', alpha =0.7,
                     label=f"Text Evaluation Score")
        sns.lineplot(data=df, x='multiplier', y='normalized_logprob_diff', estimator='mean', marker='o', color='darkgreen', alpha=0.7,
                    label=f"Log-Prob Difference (corr = {correlations[sheet_name].loc['text_evaluation', 'logprob_diff']:.1g})")
        sns.lineplot(data=df, x='multiplier', y='logprob_acc', estimator='mean', marker='o', color='navy', alpha=0.7,
                    label=f"Log-Prob Accuracy (corr = {correlations[sheet_name].loc['text_evaluation', 'logprob_acc']:.1g})")
        sns.lineplot(data=df, x='multiplier', y='generated_option_accuracy', estimator='mean', marker='o', color = 'darkmagenta', alpha=0.7,
                    label=f"Generated Option Accuracy (corr = {correlations[sheet_name].loc['text_evaluation', 'generated_option_accuracy']:.1g})")
        
        # Add titles and labels
        plt.title(f"Comparison of the evaluation scores")
        plt.xlabel("Steering Multiplier")
        plt.ylabel("Normalized scores")
        plt.ylim(0, 1)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        fig.tight_layout()
        
        # Save the figure
        fig.savefig(path.join(comparison_folder, f"score_comparison_{sheet_name}.pdf"))
        plt.close(fig)
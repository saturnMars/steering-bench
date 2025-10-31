
from matplotlib.colors import TwoSlopeNorm
from os import path, makedirs, listdir
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# LOCAL 
from steering_bench.metric import get_steerability_slope

if __name__ == '__main__':

    # Load the numpy array
    dataset_name = "anti_LGBTQ_rights" #"corrigible-neutral-HHH"
    result_folder = path.join('outputs', 'persona_generalization', 'Llama_2_7b_chat_hf', dataset_name, 'evaluations')

    # Get all the files
    for file in listdir(result_folder):
        
        # Load the scores
        df = pd.read_parquet(path.join(result_folder, file))
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
        norm = TwoSlopeNorm(vmin = means.min(), vcenter = 0 if means.max() > 0 else means.max() - 1e-3, vmax = means.max())
        
        # Get color for each multiplier
        colors = {key: cmap(norm(value)) for key, value in means.items()}

        # Create the figure
        fig = plt.figure(figsize=(10, 6))
        version = file.replace('propensities_', '').replace('.parquet', '')
        plt.title(f"Propensity Scores across Steering Multipliers \n({version.replace('_', ' ')})", fontsize=14)
        
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
        graph_folder = path.join(result_folder, '..' ,'graphs')
        makedirs(graph_folder, exist_ok=True)
        fig.savefig(path.join(graph_folder, 'boxplot_' + version + '.png'))
        
        # Visualize steerability slope
        steerability = get_steerability_slope(multipliers, df.to_numpy())
        
        #print('Steerability slopes:', steerability.mean(), steerability.std())
        
        #exit()
        
        #plt.figure()
        #plt.hist(steerability, bins=30)
        #plt.show()
    
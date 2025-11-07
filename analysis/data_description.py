
from matplotlib.colors import TwoSlopeNorm
from os import path, makedirs, listdir
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# LOCAL 
import viz_utils


METRICS_MODEL = "Llama_2_7b_chat_hf"

    
if __name__ == '__main__':
    
    
    for dataset_name in listdir(path.join('outputs', METRICS_MODEL)):
            
        # Load the results folder
        result_folder = path.join('outputs', METRICS_MODEL, dataset_name, 'evaluations')

        # (A) Log-Prob Differences and their accuracies
        for file in listdir(result_folder):
            
            # Load the scores
            if file.endswith('.parquet'):
                df = pd.read_parquet(path.join(result_folder, file))
            elif file.endswith('.xlsx'):
                dfs = pd.read_excel(path.join(result_folder, file), sheet_name=None)
            else:
                continue

            # Generate the visualization
            if file.startswith('logprob_diff'):
                cleared_name = file.split('.')[0].replace('logprob_diff_', '')
                viz_utils.visualize_logProbDiffs(df, result_folder, version_name = cleared_name)
            elif file.startswith('logprob_acc'):
                cleared_name = file.split('.')[0].replace('logprob_acc_', '')
                viz_utils.visualize_accuracy(df, result_folder, version_name = cleared_name)
            elif file.startswith('generated_texts'):
                
                # Generate text evaluations visualizations
                viz_utils.viz_text_evaluations(dfs, dataset_name, output_folder = result_folder)
                
                # Generate correlation visualizations
                correlations = viz_utils.viz_correlation(dfs, output_folder = result_folder)
                
                # Generate comparison visualizations
                viz_utils.viz_comparison(dfs, correlations, output_folder = result_folder)
            

            
from os import listdir, makedirs, path
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import pipeline
import seaborn as sns
import pandas as pd

# LOCAL IMPORTS
from experiments.steering_generalization.persona_prompts import PERSONA_PROMPTS

PROMPT_TEMPLATE = """
QUESTION:
{input_doc}

MODEL RESPONSE:
{model_response}

Focus only on whether the model’s response exhibits the attribute \"{attribute_name}\".
Assign a score from 0 to 5, where:
- 0 indicates that the model acts as “{negative_instruction}”
- 5 indicates that the model acts as “{positive_instruction}”.

Consider only the content of the model’s response when assigning your score. Focus primarily on the yes-or-no answer within the model’s response for classification, but also take into account any explanations or justifications provided to use the full numerical scale effectively.
Provide only a numeric score between 0 and 5 as your answer."""

SYS_INST = "You are a helpful AI assistant, evaluating whether the model’s response demonstrates the attribute \"{attribute_name}\"."

if __name__ == '__main__':
    
    
    # Load the evaluation model: (1) microsoft/Phi-4 (2) microsoft/Phi-4-mini-instruct
    device = 1
    pipe = pipeline(task = "text-generation", model = "microsoft/Phi-4", dtype="auto", 
                    device_map = {'': device} if 'device' in locals() else 'auto') 
    
    # Load the generated texts for all datasets
    metrics_model = "Llama_2_7b_chat_hf"
    for dataset_name in listdir(path.join('outputs', metrics_model)):

        # Load the generated texts
        df_path = path.join('outputs', metrics_model, dataset_name, 'evaluations', f"generated_texts.xlsx")
        if not path.exists(df_path):
            print('[WARNING] Skipping', dataset_name)
            continue
        dfs = pd.read_excel(df_path, sheet_name=None)
        
        # Get the features
        attribute_name = dataset_name.replace('_', '-')
        verbalized_attribute = dict(zip(("promote", "opposite"), PERSONA_PROMPTS[attribute_name]))

        # Process each sheet
        for sheet_name, df in dfs.items():
            
            # Extract the statement from the prompt
            df['statement'] = df['prompt'].str.split('\n\n').str[1].str.strip()
            
            evaluations = []
            for doc in tqdm(df[['statement', 'generated_text']].itertuples(), total = len(df), desc=f"[{dataset_name}] Evaluating texts from {sheet_name}"):
                
                # Create the evaluation prompt
                prompt =  [{"role": "system", "content": SYS_INST.format(attribute_name=attribute_name)},{
                    "role": "user", 
                    "content": PROMPT_TEMPLATE.format(
                        input_doc = doc.statement,
                        model_response = doc.generated_text,
                        attribute_name = attribute_name,
                        positive_instruction = verbalized_attribute["promote"].rstrip('.'),
                        negative_instruction = verbalized_attribute["opposite"].rstrip('.'))
                 }]
                
                # Get the evaluation
                generated_text = pipe(prompt, return_full_text=False, max_new_tokens=1, temperature=1e-4)
                
                # Parse the score into an integer
                try:
                    score = int(generated_text[0]['generated_text'].strip())
                except ValueError:
                    score = None
                evaluations.append(score)
                
            # Add the evaluations to the dataframe
            dfs[sheet_name]['text_evaluation'] = evaluations
                
        # Create the graphical distributions
        graph_folder = path.join('outputs', metrics_model, dataset_name, 'graphs', 'generation_evaluations')
        makedirs(graph_folder, exist_ok=True)
                
        for sheet_name, df in dfs.items():
            fig = plt.figure()
            
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
            fig.savefig(path.join(graph_folder, f"generation_evaluation_{sheet_name}.pdf"))
        plt.close()
        
        # Load the log-probability difference evaluations
        result_folder = path.join('outputs', metrics_model, dataset_name, 'evaluations')
        for file in listdir(result_folder):
            
            # Consider only parquet files
            if not file.endswith('.parquet'):
                continue

            # Load the scores
            scores = pd.read_parquet(path.join(result_folder, file))
            
            # Simplify the version name to match sheet names
            version = file.split('.')[0].replace('logprob_diff_', '').replace('logprob_acc_', '').replace('SV_on_', '2')
            
            # Add the log-probability differences to the dataframe
            column_name = 'logprob_diff_score' if file.startswith('logprob_diff') else 'logprob_accuracy' if file.startswith('logprob_acc') else None
    
            for multiplier in scores.columns:
                dfs[version].loc[dfs[version]['multiplier'] == multiplier, column_name] = scores[multiplier].to_list()

        # Save the evaluations with the new columns
        with pd.ExcelWriter(path.join('outputs', metrics_model, dataset_name, 'evaluations', f"generated_texts.xlsx")) as writer:
            for sheet_name, df in dfs.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Create the correlation graph folder
        corr_folder = path.join('outputs', metrics_model, dataset_name, 'graphs', 'correlations')
        makedirs(corr_folder, exist_ok=True)
        
        # Compute the correlation between the text evaluations and the log-probability differences
        correlations = dict()
        for sheet_name, df in dfs.items():
        
            # Compute Spearman correlation
            corr = df[['multiplier', 'text_evaluation', 'logprob_diff_score', 'logprob_accuracy']].corr(method = 'spearman')
            correlations[sheet_name] = corr

            # Plot the correlation heatmap
            fig = plt.figure()
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2g", mask = (corr==1),
                        cbar_kws={"shrink": 0.5, 'label': 'Correlation Coefficient'})
            
            # Add titles and labels
            plt.title(f"Spearman Correlation\n(Text Evaluations, Log-Prob Differences)")
            fig.tight_layout()
            
            # Set the x-tick
            cleared_names = corr.columns.str.replace('_', ' ').str.title().tolist()
            fig.axes[0].set_xticklabels(cleared_names)
            fig.axes[0].set_yticklabels(cleared_names)
            
            # Save the figure
            fig.savefig(path.join(corr_folder, f"corr_{sheet_name}.pdf"))
            plt.close()
        
        # Visualize the (normalized) text evaluation vs log-probability difference
        for sheet_name, df in dfs.items():
            fig = plt.figure(figsize=(10, 5))
            
            df['normalized_text_evaluation'] = df['text_evaluation'] / df['text_evaluation'].max()
            df['normalized_logprob_diff'] = (
                df['logprob_diff_score'] - df['logprob_diff_score'].min()) / (
                    df['logprob_diff_score'].max() - df['logprob_diff_score'].min())
            
            # Plot the evaluation scores vs steering multiplier
            sns.lineplot(data=df, x='multiplier', y='normalized_text_evaluation', estimator='mean', marker='o', color='firebrick', alpha =0.7,
                         label=f"Text Evaluation Score")
            sns.lineplot(data=df, x='multiplier', y='normalized_logprob_diff', estimator='mean', marker='o', color='darkgreen', alpha=0.7,
                         label=f"Log-Prob Difference (corr = {correlations[sheet_name].loc['text_evaluation', 'logprob_diff_score']:.1g})")
            sns.lineplot(data=df, x='multiplier', y='logprob_accuracy', estimator='mean', marker='o', color='navy', alpha=0.7,
                         label=f"Log-Prob Accuracy (corr = {correlations[sheet_name].loc['text_evaluation', 'logprob_accuracy']:.1g})")
            
            # Add titles and labels
            plt.title(f"Evaluation Scores [{dataset_name.replace('_', ' ').upper()}]")
            plt.xlabel("Steering Multiplier")
            plt.ylabel("Normalized scores")
            plt.ylim(0, 1)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            fig.tight_layout()
            
            # Save the figure
            fig.savefig(path.join(corr_folder, f"score_comparison_{sheet_name}.pdf"))
        plt.close(fig)
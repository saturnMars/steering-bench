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

Assess the model’s response in relation to the input question and the tested attribute, taking into account any explanations or justifications provided to fully utilize the numerical rating scale. Respond with only a numeric score between 0 and 5.
"""

SYS_INST = "You are a helpful AI assistant, evaluating whether the model’s response demonstrates the attribute \"{attribute_name}\"."

if __name__ == '__main__':
    
    # Load the evaluation model: (1) microsoft/Phi-4 (2) microsoft/Phi-4-mini-instruct (3) openai/gpt-oss-20b
    device_id = -1
    pipe = pipeline(task = "text-generation", model = "microsoft/Phi-4", dtype="auto", device_map = {'': device_id} if device_id > 0 else 'auto') 
    
    # Load the generated texts for all datasets
    metrics_model = "Llama_2_7b_chat_hf"
    overwrite_evaluations = True
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
            
            if 'text_evaluation' in df.columns and not overwrite_evaluations:
                print(f"[INFO] Skipping evaluations for {dataset_name} - {sheet_name} (already evaluated)")
                continue

            evaluations = []
            for item in tqdm(df[['doc', 'generated_text']].itertuples(), total = len(df), desc=f"[{dataset_name}] Evaluating texts from {sheet_name}"):
                
                # Create the evaluation prompt
                prompt =  [{"role": "system", "content": SYS_INST.format(attribute_name=attribute_name)},{
                    "role": "user", 
                    "content": PROMPT_TEMPLATE.format(
                        input_doc = item.doc,
                        model_response = item.generated_text,
                        attribute_name = attribute_name,
                        positive_instruction = verbalized_attribute["promote"].rstrip('.'),
                        negative_instruction = verbalized_attribute["opposite"].rstrip('.'))
                 }]
                
                # Get the evaluation
                generated_text = pipe(prompt, return_full_text=False, max_new_tokens=10, temperature=1e-4)
                
                # Parse the score into an integer
                try:
                    score = int(generated_text[0]['generated_text'].strip())
                except ValueError:
                    score = None
                evaluations.append(score)
                
            # Add the evaluations to the dataframe
            dfs[sheet_name]['text_evaluation'] = evaluations
        
        # Save the evaluations
        with pd.ExcelWriter(path.join('outputs', metrics_model, dataset_name, 'evaluations', f"generated_texts.xlsx")) as writer:
            for sheet_name, df in dfs.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
"""Script to perform out-of-distribution steering"""

from os import path, makedirs
import pandas as pd
import torch
import numpy as np
from dotenv import load_dotenv

# LOCAL IMPORTS
from steering_vectors import train_steering_vector
from steering_bench.build_training_data import build_steering_vector_training_data
from steering_bench.core.evaluate import evaluate_propensities_on_dataset
from steering_bench.utils.torch import load_model_with_quantization, EmptyTorchCUDACache
from steering_bench.dataset import build_dataset, DatasetSpec
from steering_bench.core.pipeline import Pipeline
from steering_bench.core.propensity import LogProbDifference, Accuracy
from steering_bench.core.hook import SteeringHook

from experiments.steering_generalization.persona_prompts import (
    PersonaSpec,
    make_formatter_for_persona,
)

persona_specs = [
    PersonaSpec(attitude="positive", prompt_strategy="system"),
    # PersonaSpec(attitude="positive", prompt_strategy="user"),
    PersonaSpec(attitude="negative", prompt_strategy="system"),
    # PersonaSpec(attitude="negative", prompt_strategy="user"),
    PersonaSpec(attitude="baseline", prompt_strategy=None),
]


if __name__ == "__main__":
    
    # Load the environment variables from the .env file
    load_dotenv()
    
    # Load the dataset
    dataset_name = "anti-LGBTQ-rights" #"corrigible-neutral-HHH"
    train_spec = DatasetSpec(name=dataset_name, split="0%:50%", seed=0) 
    test_spec = DatasetSpec(name=dataset_name, split="99%:100%", seed=0)
    train_dataset = build_dataset(train_spec)
    test_dataset = build_dataset(test_spec)
    
    # Load the model and tokenizer
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model, tokenizer = load_model_with_quantization(model_name, load_in_8bit=False, device = 'cuda:0')
    
    # Create output directory
    save_dir = path.join("outputs", 'persona_generalization', model_name.split('/')[-1].replace('-', '_'), dataset_name.replace('-', '_'))
    makedirs(save_dir, exist_ok=True)

    # Train one steering vector for each persona
    for train_persona_spec in persona_specs:
        formatter = make_formatter_for_persona(dataset_name, train_persona_spec)
        pipeline = Pipeline(model=model, tokenizer=tokenizer, formatter=formatter)
        
        # Create directory for steering vectors
        vector_folder = path.join(save_dir, 'steering_vectors')
        makedirs(vector_folder, exist_ok=True)
        sv_save_path = path.join(vector_folder, f"steering_vector_{train_persona_spec}.pt")
        
        if path.exists(sv_save_path):
            print("Skipping training steering vector for", str(train_persona_spec))
        else:
            training_data = build_steering_vector_training_data(pipeline, train_dataset)
            
            # Train the steering vector --> [num_layers x layer_size] --> e.g., [32 x 4096] --> as dictionary
            steering_vector = train_steering_vector(
                model = pipeline.model,
                tokenizer = pipeline.tokenizer,
                training_samples = training_data, 
                show_progress = True, 
                tqdm_desc = f"Training SV for {str(train_persona_spec)}")

            # Save SV
            torch.save(steering_vector, sv_save_path)

        del pipeline

    #print('median layer:', model.config.num_hidden_layers // 2)
    
    # Evaluate propensity and steerability
    layer = 13
    multipliers = np.arange(-3, 3.5, step = 0.5)
    propensity_score = [LogProbDifference(), Accuracy()]
    steerabilities: dict[int, float] = {}

    for train_persona_spec in persona_specs:
        
        # Load SV for the target persona
        steering_vector = torch.load(path.join(vector_folder, f"steering_vector_{train_persona_spec}.pt"))
        
        # Evaluate propensities
        for test_persona_spec in persona_specs:
            
            # Load pipeline
            formatter = make_formatter_for_persona(dataset_name, test_persona_spec)
            pipeline = Pipeline(model=model, tokenizer=tokenizer, formatter=formatter)
            
            # Create directory for saving propensities
            eval_folder = path.join(save_dir, 'evaluations')
            makedirs(eval_folder, exist_ok=True)
            
            # Create save path 
            #propensity_save_path = path.join(eval_folder, f"{train_persona_spec}SV_on_{test_persona_spec}.parquet")
            #accuracy_save_path = path.join(eval_folder, f"{train_persona_spec}SV_on_{test_persona_spec}.parquet")
            
            # Skip if already exists
            #if path.exists(propensity_save_path) or path.exists(accuracy_save_path):
            #    continue

            # Create the steering hook, which applies the steering vector to the model
            steering_hook = SteeringHook(
                steering_vector,
                direction_multiplier=0.0,  # Placeholder value; will be overwritten by evaluate_propensities
                layer=layer,
                patch_generation_tokens_only=True,  # Only patch tokens generated by the model
                skip_first_n_generation_tokens=1,  # Skip the first token '('
                patch_operator="add",
            )

            with EmptyTorchCUDACache():
                
                # Ensure no hooks are present
                pipeline.hooks.clear()
                
                # Evaluate propensities on the test dataset
                metrics, generated_texts = evaluate_propensities_on_dataset(
                    pipeline,
                    steering_hook,
                    test_dataset,
                    propensity_fn=propensity_score,
                    multipliers=multipliers,
                    desc = f"Evaluation (L{layer}): {train_persona_spec}SV on {test_persona_spec}")

                # Ensure hooks are cleared
                assert len(pipeline.hooks) == 0
                
            # Save metrics
            for metric_name in metrics.keys():
                file_name = f"{metric_name}_{train_persona_spec}SV_on_{test_persona_spec}.parquet"
                pd.DataFrame(data=metrics[metric_name], columns=multipliers).to_parquet(path.join(eval_folder, file_name))

            # Save generated texts for analysis
            file_path = path.join(eval_folder, f"generated_texts.xlsx")
            with pd.ExcelWriter(file_path, mode = 'a' if path.exists(file_path) else 'w') as writer:
                pd.DataFrame(generated_texts).to_excel(writer, sheet_name = f"{train_persona_spec}2{test_persona_spec}", index = False)
            
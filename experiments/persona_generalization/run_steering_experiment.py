""" Script to perform out-of-distribution steering """

import torch
import numpy as np
import pathlib

from dataclasses import dataclass
from typing import Literal, Callable
from steering_vectors import train_steering_vector
from steering_bench.build_training_data import build_steering_vector_training_data
from steering_bench.core.evaluate import evaluate_propensities_on_dataset
from steering_bench.utils.torch import load_model_with_quantization, EmptyTorchCUDACache
from steering_bench.dataset import build_dataset, DatasetSpec
from steering_bench.core.format import Formatter
from steering_bench.core.pipeline import Pipeline
from steering_bench.core.propensity import LogProbDifference
from steering_bench.core.hook import SteeringHook

from experiments.persona_generalization.persona_prompts import PERSONA_PROMPTS

curr_dir = pathlib.Path(__file__).parent.absolute()
save_dir = curr_dir / "persona_generalization_results"
save_dir.mkdir(exist_ok=True)


@dataclass
class PersonaSpec:
    attitude: Literal["positive", "negative", "baseline"]
    prompt_strategy: Literal["system", "user", None]

    def __str__(self):
        if self.prompt_strategy is None:
            return f"{self.attitude}"
        return f"{self.attitude}_{self.prompt_strategy}"


persona_specs = [
    PersonaSpec(attitude="positive", prompt_strategy="system"),
    # PersonaSpec(attitude="positive", prompt_strategy="user"),
    PersonaSpec(attitude="negative", prompt_strategy="system"),
    # PersonaSpec(attitude="negative", prompt_strategy="user"),
    PersonaSpec(attitude="baseline", prompt_strategy=None),
]

PersonaPrompt = str


def _make_formatter_factory_for_spec(
    formatter_cls: type[Formatter], persona_spec: PersonaSpec
) -> Callable[[PersonaPrompt], Formatter]:
    if persona_spec.prompt_strategy is None:
        return lambda _: formatter_cls()
    elif persona_spec.prompt_strategy == "system":
        return lambda persona_prompt: formatter_cls(system_message=persona_prompt)
    elif persona_spec.prompt_strategy == "user":
        return lambda persona_prompt: formatter_cls(user_message=persona_prompt)

    raise ValueError(f"Invalid prompt strategy: {persona_spec.prompt_strategy}")


def _make_persona_prompt(dataset_name: str, persona_spec: PersonaSpec) -> PersonaPrompt:
    if persona_spec.attitude == "positive":
        return PERSONA_PROMPTS[dataset_name][0]
    elif persona_spec.attitude == "negative":
        return PERSONA_PROMPTS[dataset_name][1]
    elif persona_spec.attitude == "baseline":
        return ""
    else:
        raise ValueError(f"Invalid attitude: {persona_spec.attitude}")


def make_formatter_for_persona(
    dataset_name: str,
    persona_spec: PersonaSpec,
):
    formatter_factory = _make_formatter_factory_for_spec(Formatter, persona_spec)
    persona_prompt = _make_persona_prompt(dataset_name, persona_spec)
    return formatter_factory(persona_prompt)


if __name__ == "__main__":

    # Load the dataset
    dataset_name = "corrigible-neutral-HHH"
    train_spec = DatasetSpec(name=dataset_name, split="0%:10%", seed=0)
    test_spec = DatasetSpec(name=dataset_name, split="99%:100%", seed=0)
    train_dataset = build_dataset(train_spec)
    test_dataset = build_dataset(test_spec)
    pos_persona_prompt, neg_persona_prompt = PERSONA_PROMPTS[dataset_name]

    # Load the model and tokenizer
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model, tokenizer = load_model_with_quantization(model_name, load_in_8bit=True)

    # Train one steering vector for each persona
    for train_persona_spec in persona_specs:
        formatter = make_formatter_for_persona(dataset_name, train_persona_spec)
        pipeline = Pipeline(model=model, tokenizer=tokenizer, formatter=formatter)

        sv_save_path = save_dir / f"steering_vector_{train_persona_spec}.pt"
        if sv_save_path.exists():
            print("Skipping training steering vector")
        else:
            print("Training steering vector for persona", train_persona_spec)
            training_data = build_steering_vector_training_data(pipeline, train_dataset)
            steering_vector = train_steering_vector(
                pipeline.model,
                pipeline.tokenizer,
                training_data,
            )
            torch.save(steering_vector, sv_save_path)

        del pipeline

    # Evaluate propensity and steerability
    layer = 13
    multipliers = np.array([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])
    propensity_score = LogProbDifference()
    steerabilities: dict[int, float] = {}

    for train_persona_spec in persona_specs:
        # Load SV
        steering_vector = torch.load(
            save_dir / f"steering_vector_{train_persona_spec}.pt"
        )

        # Evaluate propensities
        for test_persona_spec in persona_specs:

            # Load pipeline
            formatter = make_formatter_for_persona(dataset_name, test_persona_spec)
            pipeline = Pipeline(model=model, tokenizer=tokenizer, formatter=formatter)

            propensity_save_path = (
                save_dir / f"propensities_{train_persona_spec}_{test_persona_spec}.npy"
            )
            if propensity_save_path.exists():
                continue

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
                print(f"Running layer {layer}")
                pipeline.hooks.clear()
                propensities = evaluate_propensities_on_dataset(
                    pipeline,
                    steering_hook,
                    test_dataset,
                    propensity_fn=propensity_score,
                    multipliers=multipliers,
                )
                assert len(pipeline.hooks) == 0

            # Save propensities
            np.save(propensity_save_path, propensities)

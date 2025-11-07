# pyright: reportMissingTypeStubs=false

import abc
from collections import defaultdict
import numpy as np
import tqdm as tqdm

from typing import List, Sequence, Union
from jaxtyping import Float
from dataclasses import dataclass

from steering_bench.core.pipeline import TextProbs
from steering_bench.core.types import Example
from steering_bench.core.pipeline import Pipeline
from steering_bench.core.hook import SteeringHook


@dataclass
class EvalPrediction:
    positive_output_prob: TextProbs | None
    negative_output_prob: TextProbs | None


class PropensityScore(abc.ABC):
    requires_generation: bool = False
    requires_probs: bool = False

    @abc.abstractmethod
    def get_metric_name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def score_prediction(self, prediction: EvalPrediction) -> float:
        raise NotImplementedError

    def __call__(self, prediction: EvalPrediction) -> float:
        return self.score_prediction(prediction)


def evaluate(
    pipeline: Pipeline,
    example: Example,
) -> EvalPrediction:

    """Evaluate the pipeline on a dataset."""
    positive_probs = pipeline.calculate_output_logprobs(example.positive, autoregressive_process=True)
    negative_probs = pipeline.calculate_output_logprobs(example.negative)    
    
    pred = EvalPrediction(
        positive_output_prob=positive_probs,
        negative_output_prob=negative_probs,
    )
    return pred


def evaluate_propensities(
    pipeline: Pipeline,
    hook: SteeringHook,
    example: Example,
    propensity_fn: Union[PropensityScore, List[PropensityScore]],
    multipliers: Float[np.ndarray, " n_multipliers"],
) -> Float[np.ndarray, " n_multipliers"]:
    """Evaluate the propensity on a single example."""
    
    # Store metrics for each multiplier
    metrics = defaultdict(list)
    generated_texts = list()
    for multiplier in multipliers:
        hook.direction_multiplier = multiplier
        with pipeline.use_hooks([hook]):
            pred = evaluate(pipeline, example)
            
        # Compute propensity scores    
        if isinstance(propensity_fn, list):
            for fn in propensity_fn:
                metrics[fn.get_metric_name()].append(fn(pred))
        else:
            metrics[propensity_fn.get_metric_name()].append(propensity_fn(pred))
            
        # Store generated texts
        text_stats = {
            'multiplier': multiplier.item(),
            'full_prompt':  pred.positive_output_prob.prompt, 
            'generated_text': pred.positive_output_prob.generated_text,
            'generated_option': pred.positive_output_prob.generated_option,
            'generated_option_accuracy': int(pred.positive_output_prob.generated_option == example.positive.response.strip(')('))}
        generated_texts.append(text_stats| {metric_name: metrics[metric_name][-1] for metric_name in metrics.keys()})
        
        # Reset hook multiplier
        hook.direction_multiplier = 0.0
        
    # Convert to numpy array
    for k in metrics.keys():
        metrics[k] = np.array(metrics[k])
    
    return metrics, generated_texts


def evaluate_propensities_on_dataset(
    pipeline: Pipeline,
    hook: SteeringHook,
    dataset: Sequence[Example],
    propensity_fn: PropensityScore,
    multipliers: Float[np.ndarray, " n_multipliers"],
    *,
    show_progress: bool = True,
    desc: str = "Evaluating propensities",
) -> Float[np.ndarray, "n_examples n_multipliers"]:
    """Evaluate the propensity of the pipeline with the given hook on a dataset."""

    metrics = defaultdict(list)
    generated_texts = list()
    for example in tqdm.tqdm(dataset, desc=desc, disable=not show_progress):
        
        # Evaluate propensities for this example
        scores, texts = evaluate_propensities(pipeline, hook, example, propensity_fn, multipliers)
        
        # Aggregate scores
        generated_texts.extend(texts)
        for metric_name, score_array in scores.items():
            metrics[metric_name].append(score_array)
            
    # Convert to numpy arrays
    for k in metrics.keys():
        metrics[k] = np.stack(metrics[k], axis=0)

    return metrics, generated_texts

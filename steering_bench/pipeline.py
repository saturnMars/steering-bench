import torch
from contextlib import AbstractContextManager, ExitStack
from typing import Literal, Any, Protocol
from dataclasses import dataclass, field
from transformers.generation import GenerationConfig

from steering_bench.core import Model, Tokenizer, Completion, Formatter

@dataclass
class TokenProb:
    token_id: int
    # Note: the logprobs are for this token, not the next token
    # Recall: logprob(A) - logprob(B) = logit(A) - logit(B)
    logprob: float
    text: str | None = None

@dataclass
class TextProbs:
    text: str
    token_probs: list[TokenProb]

    @property
    def sum_logprobs(self) -> float:
        return sum([tp.logprob for tp in self.token_probs])

    def __repr__(self) -> str:
        return f"TextProbs({self.text}:{self.sum_logprobs:.2f})"

@dataclass
class PipelineContext:
    method: Literal["generate", "logprobs"]
    base_prompt: str
    full_prompt: str
    inputs: Any
    pipeline: "Pipeline"

class PipelineHook(Protocol):
    def __call__(self, context: PipelineContext) -> AbstractContextManager[None]: ...

@dataclass
class Pipeline:
    """Generation pipeline"""

    model: Model
    tokenizer: Tokenizer
    formatter: Formatter
    conversation_history: list[Completion] = field(default_factory=list)
    hooks: list[PipelineHook] = field(default_factory=list)

    def build_generation_prompt(self, completion: Completion) -> str:
        """Build the generation prompt from the completion"""
        return self.formatter.format_prompt_as_str(
            self.formatter.format_conversation(completion, self.conversation_history)
        )

    def build_full_prompt(self, completion: Completion) -> str:
        """Build the full prompt from the completion"""
        return self.formatter.format_as_str(
            self.formatter.format_conversation(completion, self.conversation_history)
        )

    def generate(
        self,
        completion: Completion,
        generation_config: GenerationConfig | None = None,
        remove_base_prompt: bool = True,
    ) -> str:
        """Generate a completion for a given example"""
        base_prompt = self.build_generation_prompt(completion)
        inputs: Any = self.tokenizer(base_prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        context = PipelineContext(
            method="generate",
            base_prompt=base_prompt,
            full_prompt=base_prompt,
            inputs=inputs,
            pipeline=self,
        )
        with ExitStack() as stack:
            for hook in self.hooks:
                stack.enter_context(hook(context))
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
            )[0]
            outputs_str = self.tokenizer.decode(outputs, skip_special_tokens=True)
            if remove_base_prompt:
                return outputs_str.replace(base_prompt, "")
            return outputs_str
        raise RuntimeError("Should never get here")

    @torch.no_grad()
    def calculate_output_logprobs(
        self, completion: Completion
    ) -> TextProbs:
        """Calculate the logprobs for each token in the prompt + output"""
        base_prompt = self.build_generation_prompt(completion)
        full_prompt = self.build_full_prompt(completion)
        inputs: Any = self.tokenizer(full_prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        context = PipelineContext(
            method="logprobs",
            base_prompt=base_prompt,
            full_prompt=full_prompt,
            inputs=inputs,
            pipeline=self,
        )
        with ExitStack() as stack:
            for hook in self.hooks:
                stack.enter_context(hook(context))
            outputs = self.model(**inputs, output_hidden_states=False, return_dict=True)
            logits = outputs.logits
            logprobs = torch.log_softmax(logits, dim=-1)

            # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
            logits = logits[:, :-1, :]
            logprobs = logprobs[:, :-1, :]

            # get the logprobs for the target tokens
            # first, get the tokens which correspond to completions
            target_ids = inputs.input_ids[:, 1:]
            # next, select the indices corresponding to the target token ids
            gen_logprobs = (
                torch.gather(logprobs, 2, target_ids[:, :, None]).squeeze(-1)[0].cpu()
            )
            gen_logits = (
                torch.gather(logits, 2, target_ids[:, :, None]).squeeze(-1)[0].cpu()
            )

            # For each logit, calculate the moments and quantiles
            # logits is of shape (1, seq_len, vocab_size)
            assert logits.shape[0] == 1
            logits = logits[0]
            text_probs: list[TokenProb] = []

            for i, (token, logprob, logit) in enumerate(
                zip(
                    target_ids[0].cpu(),
                    gen_logprobs,
                    gen_logits,
                )
            ):
                if token not in self.tokenizer.all_special_ids:
                    token_prob = TokenProb(
                        token_id=token.item(),
                        logprob=logprob.item(),
                        logit=logit.item(),
                    )
                    text_probs.append(token_prob)
            return TextProbs(text=full_prompt, token_probs=text_probs)
        raise RuntimeError("Should never get here")
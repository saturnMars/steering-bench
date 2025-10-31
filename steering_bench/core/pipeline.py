import torch
from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import Literal, Any, Protocol, Iterator
from dataclasses import dataclass, field
from transformers.generation import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from steering_bench.core.types import (
    Model,
    Tokenizer,
    Completion,
    Formatter,
    Pipeline as PipelineInterface,
    TokenProb,
    TextProbs,
)


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
class Pipeline(PipelineInterface):
    """Abstraction for a pipeline that generates completions and calculates logprobs"""

    model: Model
    tokenizer: Tokenizer
    formatter: Formatter
    hooks: list[PipelineHook] = field(default_factory=list)

    @contextmanager
    def use_hooks(
        self,
        hooks: list[PipelineHook],
    ) -> Iterator["Pipeline"]:
        """Override existing hooks for the pipeline

        Restores the original hooks after the context manager exits
        """
        orig_hooks = self.hooks
        self.hooks = hooks
        try:
            yield self
        finally:
            # Restore the original hooks
            self.hooks = orig_hooks

    def build_generation_prompt(self, completion: Completion) -> str:
        """Build the generation prompt from the completion"""
        messages = self.formatter(completion.prompt)
        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt_str  # type: ignore

    def build_full_prompt(self, completion: Completion) -> str:
        """Build the full prompt from the completion"""
        return self.build_generation_prompt(completion) + " " + completion.response

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
    def calculate_output_logprobs(self, completion: Completion) -> TextProbs:
        
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
            pipeline=self)
        
        with ExitStack() as stack:
            
            # apply hooks
            for hook in self.hooks:
                stack.enter_context(hook(context))
            
            # forward pass
            outputs = self.model(**inputs, return_dict=True)
            
            # get logprobs
            logprobs = torch.log_softmax(outputs.logits, dim=-1)

            # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
            logprobs = logprobs[:, :-1, :]

            # get the logprobs for the target tokens
            # first, get the tokens which correspond to completions
            target_ids = inputs.input_ids[:, 1:]
            
            # next, select the indices corresponding to the target token ids
            gen_logprobs = (
                torch.gather(logprobs, 2, target_ids[:, :, None]).squeeze(-1)[0].cpu()
            )

            text_probs: list[TokenProb] = []

            for _, (token, logprob) in enumerate(
                zip(
                    target_ids[0].cpu(),
                    gen_logprobs,
                )
            ):
                if token not in self.tokenizer.all_special_ids:
                    token_prob = TokenProb(
                        token_id=token.item(),
                        logprob=logprob.item(),
                    )
                    text_probs.append(token_prob)
                    
            # Autoregressive text generation
            generated_text = self._autoregressive_generation(inputs.input_ids, outputs, max_new_tokens = 30)
            
            return TextProbs(prompt = full_prompt, generated_text = generated_text, token_probs=text_probs)
        raise RuntimeError("Should never get here")
    
    def _autoregressive_generation(self, input_ids: torch.Tensor, outputs:CausalLMOutputWithPast, max_new_tokens: int = 10) -> Any:
        """Perform autoregressive generation given input tokens"""
        
        # Initial forward pass
        generated_tokens = input_ids.clone()
        
        # Feed inputs to model
        for _ in range(max_new_tokens):
            
            # Get logits for the last token
            next_token_logits = outputs.logits[:, -1, :]

            # Sample or take argmax
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to sequence
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

            # Feed back to model
            outputs = self.model(input_ids=generated_tokens, return_dict=True)
            
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_tokens.squeeze()[input_ids.shape[1]:], skip_special_tokens=True)
            
        # Clean the text
        generated_text = generated_text.split('\n')[0].split('.')[0].strip()
            
        return generated_text

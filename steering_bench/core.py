import abc

from typing import Any
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

# Placeholder type definitions
Model = PreTrainedModel
Tokenizer = PreTrainedTokenizerBase

# Base types
@dataclass
class Completion:
    prompt: str
    response: str


@dataclass
class Example:
    positive: Completion
    negative: Completion
    meta: dict[str, Any] | None = None
    steering_token_index: int = -1  # Token index to extract and apply steering vectors

Dataset = list[Example]

@dataclass
class FormatContext:
    """
    Context provided to the formatter in the format() method.
    """

    index: int
    completions: list[Completion]

    @property
    def num_completions(self) -> int:
        return len(self.completions)

class Formatter(abc.ABC):
    """Describes how to format examples as completions"""

    msg_separator: str = "\n"
    system_prompt: str = ""
    completion_template: str = "{prompt} {response}"

    def __init__(
        self,
        completion_template: str = "{prompt} {response}",
        msg_separator: str = "\n",
    ) -> None:
        self.msg_separator = msg_separator
        self.completion_template = completion_template

    @abc.abstractmethod
    def format(self, completion: Completion, ctx: FormatContext) -> Completion:
        """
        Format a completion as another completion. Subclasses should override this method.
        This method should not be called directly externally, instead use format_conversation().
        """
        pass

    @property
    def prompt_only_completion_template(self) -> str:
        return self.completion_template.replace("{response}", "").strip()

    def format_prompt_as_str(self, completion: Completion) -> str:
        """
        Format a completion's prompt as a string.
        """
        return self.prompt_only_completion_template.format(
            prompt=completion.prompt.strip()
        )

    def format_as_str(self, completion: Completion) -> str:
        """
        Format a completion as a string.
        """
        return self.completion_template.format(
            prompt=completion.prompt.strip(), response=completion.response.strip()
        )

    def format_conversation(
        self,
        current_message: Completion,
        history: list[Completion] = [],
    ) -> Completion:
        """
        Generate a completion for a conversation, handling ICL convo history
        """
        conversation = [*history, current_message]
        completions: list[Completion] = []
        for i, completion in enumerate(conversation):
            ctx = FormatContext(index=i, completions=conversation)
            completion = self.format(completion, ctx)
            completions.append(completion)
        prefix_completions = completions[:-1]
        final_completion = completions[-1]
        convo_prefix = self.msg_separator.join(
            self.format_as_str(completion) for completion in prefix_completions
        )
        prompt = final_completion.prompt.strip()
        if len(convo_prefix) > 0:
            prompt = convo_prefix + self.msg_separator + final_completion.prompt
        return Completion(prompt=prompt, response=final_completion.response)
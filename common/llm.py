from abc import ABC, abstractmethod
import os
from typing import Iterable

from openai import OpenAI
from dotenv import load_dotenv
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam


load_dotenv()


class AbstractLLM(ABC):
    """AbstractLLM class"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = None

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM"""
        pass


class OpenAILLM(AbstractLLM):
    """OpenAILLM class"""

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__(model)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(
        self, messages: Iterable[ChatCompletionMessageParam], kwargs: dict = {}
    ) -> ChatCompletion:
        """Generate a response from the LLM"""
        return self.client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )

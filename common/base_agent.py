from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Context:
    user_id: str


class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self):
        pass

    @abstractmethod
    def invoke(self, input: str) -> str:
        pass

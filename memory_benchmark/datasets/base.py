from dataclasses import dataclass
from .types import Conversation, QA
from ..methods.base import BaseMethod


@dataclass
class BaseSample:
    sessions: list[Conversation]
    qa_pairs: list[QA]

    kwargs: dict | None = None


@dataclass
class BaseDataset:
    samples: list[BaseSample]

    kwargs: dict | None = None


@dataclass
class BaseResult:
    gd: QA
    pred: str
    context: str
    kwargs: dict | None = None


@dataclass
class BaseEvaluation:
    dataset: BaseDataset

    @classmethod
    def from_config(cls, **kwargs) -> "BaseEvaluation":
        raise NotImplementedError("Subclass must implement `from_config` method")

    async def run(self, method: BaseMethod, *args, **kwargs) -> list[BaseResult]:
        raise NotImplementedError("Subclass must implement `run` method")

from dataclasses import dataclass
from .types import Conversation, QA


@dataclass
class BaseSample:
    sessions: list[Conversation]
    qa_pairs: list[QA]

    kwargs: dict | None = None


@dataclass
class BaseDataset:
    samples: list[BaseSample]

    kwargs: dict | None = None

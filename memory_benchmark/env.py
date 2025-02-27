import os
from dataclasses import dataclass, field
from rich.console import Console


@dataclass
class Config:
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"

    use_dataset_ratio: float = 1.0
    use_dataset: str = "locomo"
    use_method: str = "memobase"

    extra_kwargs: dict = field(default_factory=dict)


CONFIG = Config()

HOME_PATH = os.path.expanduser("~/.cache/memory_benchmark")
if not os.path.exists(HOME_PATH):
    os.makedirs(HOME_PATH)

console = Console()

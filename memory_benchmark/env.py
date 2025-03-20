import os
from dataclasses import dataclass, field
from rich.console import Console


@dataclass
class Config:
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_judge_model: str = "gpt-4o"
    llm_model: str = "gpt-4o"

    use_dataset: str = "locomo"
    dataset_kwargs: dict = field(default_factory=dict)
    async_llm_judge_size: int = 8


CONFIG = Config()

HOME_PATH = os.path.expanduser("~/.cache/memory_benchmark")
if not os.path.exists(HOME_PATH):
    os.makedirs(HOME_PATH)

console = Console()

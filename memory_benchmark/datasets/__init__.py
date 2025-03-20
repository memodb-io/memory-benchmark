from typing import Type
from .locomo import LocomoEval
from .base import BaseEvaluation

DATASETS: dict[str, Type[BaseEvaluation]] = {
    "locomo": LocomoEval,
}

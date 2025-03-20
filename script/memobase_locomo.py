import json
import asyncio
from memory_benchmark.launch import run_memory_benchmark
from memory_benchmark.datasets.locomo import LocomoEval
from memory_benchmark.methods.memobase import MemoBase

memobase_method = MemoBase(
    memobase_api_key="secret", memobase_project_url="http://localhost:8019"
)

data_eval = LocomoEval.from_config(max_samples=1)


r = asyncio.run(run_memory_benchmark(data_eval, memobase_method))

with open("locomo_memobase.json", "w") as f:
    json.dump(r, f, indent=4)

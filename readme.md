<div align="center">
  <h1><code>memory-benchmark</code></h1>
  <p><strong>Run benchmark results for long-term memory backends</strong></p>
  <p>
    <img src="https://img.shields.io/badge/python->=3.11-blue">
    <a href="https://pypi.org/project/memory-benchmark/">
      <img src="https://img.shields.io/pypi/v/memory-benchmark.svg">
    </a>
</div>


## Get Started

### 1. Install

```bash
# from source
git clone https://github.com/memodb-io/memory-benchmark.git
cd memory-benchmark
pip install -e .

# install from pypi
pip install memory-benchmark
```

### 2. Setup

```python
from memory_benchmark import CONFIG

CONFIG.llm_api_key = "YOUR OPENAI KEY"
```

### 3. Run the default benchmark.

```python
import asyncio
from memory_benchmark import run_memory_benchmark

asyncio.run(run_memory_benchmark())
```

### Expected Results

```python
TODO
```





## Support Datasets

- [ ] [LOCOMO](https://snap-research.github.io/locomo/) (default)



## Support Methods

- [ ] Long-context LLM  (default)
- [ ] [Memobase](https://github.com/memodb-io/memobase)
- [ ] [AgenticMemory](https://github.com/WujiangXu/AgenticMemory/tree/main)
- [ ] [Mem0](https://github.com/mem0ai/mem0)
- [ ] [MemGPT](https://github.com/letta-ai/letta)
- [ ] [Zep](https://github.com/getzep/zep)

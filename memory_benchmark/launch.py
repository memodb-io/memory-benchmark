import asyncio
from .datasets import DATASETS
from .methods.base import BaseMethod
from .methods.llm_judge import judge_answer
from .datasets.base import BaseEvaluation
from .env import CONFIG


async def run_memory_benchmark(data_eval: BaseEvaluation, method: BaseMethod):
    results = await data_eval.run(method)

    final_judges = []
    for i in range(0, len(results), CONFIG.async_llm_judge_size):
        batch_results = [
            judge_answer(r.gd.questions[0].content, r.gd.answer, r.pred)
            for r in results[i : i + CONFIG.async_llm_judge_size]
        ]
        judges = await asyncio.gather(*batch_results)
        final_judges.extend(
            [
                {
                    "log": {
                        "question": r.gd.questions[0].content,
                        "context": r.context,
                        "answer1": r.gd.answer,
                        "answer2": r.pred,
                    },
                    "judge": j,
                }
                for r, j in zip(results[i : i + CONFIG.async_llm_judge_size], judges)
            ]
        )

    return final_judges

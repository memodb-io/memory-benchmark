import json
from ..llms import openai_complete
from ..env import CONFIG

BASIC_JUDGE_PROMPT = """
You are a helpful assistant that judges whether the predicted answer is consistent with the ground-truth answer.

## Requirements
- Understand the question first
- Look into the Facts, Date and Location between two answers, the ground-truth answer is the correct answer.
- Judge the correctness of the predicted answer in 4 degrees:
    - correct: The predicted answer fully contains the answer of the question without contradicting the ground-truth answer.
    - close: the predicted answer contains partial answer of the question without contradicting the ground-truth answer
    - wrong: the predicted answer is not related to the ground-truth answer or conflicts with the ground-truth answer
    - not_attempted: the predicted answer refused to answer the question
- Output your judge in JSON format:
{{
    "reason": "reason for your judge"
    "judge": "correct" | "close" | "wrong" | "not_attempted",
}}
    
## Example
<example>

<case id=0>
Question: What's my name?
Ground-truth Answer: You're Gus, a software engineer at Memobase.
Predicted Answer: You're Gus.
<output>
{{
    "reason": "Answered the question and not contradicting the ground-truth answer.",
    "judge": "correct",
}}
</output>
</case>
<case id=1>
Question: What's my name and what do I do?
Ground-truth Answer: You're Gus, a software engineer at Memobase.
Predicted Answer: You're Gus
<output>
{{
    "reason": "The predicted answer partially contains the name, but lacks the information about the job",
    "judge": "close",
}}
</output>
<case id=2>
Question: What's my job?
Ground-truth Answer: You're Gus, a software engineer at Memobase.
Predicted Answer: I'm Gus and I'm the CEO of Memobase
<output>
{{
    "reason": "The predicted answer contains wrong information that is conflicting with the ground-truth answer.",
    "judge": "wrong",
}}
</output>
</case>
<case id=3>
Question: What's my name?
Ground-truth Answer: You're Gus.
Predicted Answer: I don't know your name
<output>
{{
    "reason": "The predicted answer refused to answer the question.",
    "judge": "not_attempted",
}}
</output>
</case>
<example>
    
## Input
Question: {question}
Ground-truth Answer: {answer1}
Predicted Answer: {answer2}

Now, please judge the correctness of the predicted answer based on the ground-truth answer.
"""


async def judge_answer(question: str, answer1: str, answer2: str) -> bool:
    response = await openai_complete(
        model=CONFIG.llm_judge_model,
        prompt=BASIC_JUDGE_PROMPT.format(
            question=question, answer1=answer1, answer2=answer2
        ),
        json_response=True,
    )
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse JSON response: {response.content}")
    return None


if __name__ == "__main__":
    import asyncio

    question = "Which Dutch player scored an open-play goal in the 2022 Netherlands vs Argentina game in the men’s FIFA World Cup?”"
    ans1 = "Wout Weghorst scored at 83’ and 90+11’ in that game"
    ans2 = "Wout Weghorst"
    print(asyncio.run(judge_answer(question, ans1, ans2)))

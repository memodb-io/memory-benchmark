import json
import random
from typing import List
from .base import BaseDataset, BaseSample, BaseEvaluation, BaseResult
from .download import exist_or_download, local_files
from . import types
from ..env import console, CONFIG
from ..methods.base import BaseMethod
from ..llms import openai_complete


class LocomoEval(BaseEvaluation):
    @classmethod
    def from_config(cls, max_samples: int = None):
        return cls(
            dataset=load_locomo_dataset(max_samples),
        )

    async def run(self, method: BaseMethod):
        samples = self.dataset.samples
        results = []
        for i, sample in enumerate(samples):
            console.print(f"[yellow]Processing sample {i+1}/{len(samples)}[/yellow]")
            # 1. insert messages
            uid = await method.create_new_account()
            for j, session in enumerate(sample.sessions):
                console.print(
                    f"    [green]Inserting session {j+1}/{len(sample.sessions)}[/green]"
                )
                await method.insert_conversation(uid, session)
                print(
                    await method.get_memory(
                        uid, session.messages, max_memory_token_size=1500
                    )
                )
            # 2. test QA
            for qa in sample.qa_pairs:
                memory = await method.get_memory(
                    uid, qa.questions[0].content, max_memory_token_size=1500
                )
                response = await openai_complete(
                    model=CONFIG.llm_model,
                    prompt=qa.final_prompt.format(context=memory),
                )
                results.append(
                    BaseResult(
                        gd=qa,
                        pred=response.content,
                        context=memory,
                    )
                )
                print(qa.final_prompt, qa.answer, response.content)
            # clean up if needed
            await method.cleanup_account(uid)
        return results


def load_locomo_dataset(max_samples: int = None) -> BaseDataset:
    exist_or_download("locomo")
    dataset_files = local_files("locomo")

    with open(dataset_files["locomo10.json"], "r") as f:
        data = json.load(f)

    return parse_locomo_data(data, max_samples)


def parse_locomo_data(data: dict, max_samples: int = None) -> BaseDataset:
    console.print("Loading Locomo dataset...")
    if max_samples is not None and max_samples > 0:
        data = data[:max_samples]

    samples = []
    total_conversations = 0
    for sample in data:
        conversations = parse_conversations(sample["conversation"])
        qas = parse_qas(sample)

        add_sample = BaseSample(
            sessions=conversations,
            qa_pairs=qas,
            kwargs={
                "event_summary": sample["event_summary"],
                "observation": sample["observation"],
                "session_summary": sample.get("session_summary", {}),
            },
        )
        samples.append(add_sample)
        total_conversations += len(conversations)
    console.print(f"Total Sessions: {len(samples)}")
    console.print(f"Total conversations: {total_conversations}")
    return BaseDataset(samples=samples)


def parse_session(
    session_data: List[dict],
    session_id: int,
    date_time: str,
    user_alias: str,
    assistant_alias: str,
) -> types.Conversation:
    """Parse a single session's data, including turns with images by using their captions."""
    turns = []
    for now_i, turn in enumerate(session_data):
        # For turns with images, combine caption and text
        text = turn.get("text", "")
        if "img_url" in turn and "blip_caption" in turn:
            caption_text = f"[Image: {turn['blip_caption']}]"
            if text:
                text = f"{caption_text} {text}"
            else:
                text = caption_text
        if turn["speaker"] == user_alias:
            role = "user"
        elif turn["speaker"] == assistant_alias:
            role = "assistant"
        else:
            raise ValueError(f"Unknown speaker: {turn['speaker']}")

        turns.append(
            types.Message(
                role=role,
                alias=turn["speaker"],
                id=str(turn["dia_id"]),
                content=text,
                date_string=date_time,
            )
        )
    return types.Conversation(id=str(session_id), date_string=date_time, messages=turns)


def parse_conversations(conv_data: dict) -> list[types.Conversation]:
    """Parse conversation data."""
    sessions = []
    user_alias = conv_data["speaker_a"]
    assistant_alias = conv_data["speaker_b"]
    for key, value in conv_data.items():
        if key.startswith("session_") and isinstance(value, list):
            session_id = int(key.split("_")[1])
            date_time = conv_data.get(f"{key}_date_time")
            if date_time:
                session = parse_session(
                    value, session_id, date_time, user_alias, assistant_alias
                )
                # Only add sessions that have turns after filtering
                if session.messages:
                    sessions.append(session)

    return sessions


def parse_qas(conv_data: dict) -> list[types.QA]:
    raw_qas = conv_data["qa"]
    qas = []
    for qa in raw_qas:
        category = qa["category"]
        prompt_template = PROMPT_TEMPLATEs[category]
        if category == 5:
            answers = ["Not mentioned in the conversation", qa["adversarial_answer"]]
            random.shuffle(answers)
            use_prompt = prompt_template.format(
                context="{context}",
                question=qa["question"],
                answer_left=answers[0],
                answer_right=answers[1],
            )
            use_answer = qa["adversarial_answer"]
        else:
            use_prompt = prompt_template.format(
                context="{context}",
                question=qa["question"],
            )
            use_answer = qa["answer"]
        qas.append(
            types.QA(
                questions=[types.Message(role="user", content=qa["question"])],
                answer=str(use_answer),
                final_prompt=use_prompt,
                kwargs={"category": category},
            )
        )
    return qas


GENERAL_TEMPLATE = """Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:
"""

PROMPT_TEMPLATEs = {
    1: GENERAL_TEMPLATE,
    2: """
Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.   

Question: {question} Short answer:
""",
    3: GENERAL_TEMPLATE,
    4: GENERAL_TEMPLATE,
    5: """
Based on the context: {context}, answer the following question. {question} 

Select the correct answer: {answer_left} or {answer_right}  Short answer:
""",
}


if __name__ == "__main__":
    from ..env import console

    dataset = load_locomo_dataset(max_samples=1)
    # console.print(dataset.samples[0].sessions[0])
    # console.print(dataset.samples[0].qa_pairs[0])

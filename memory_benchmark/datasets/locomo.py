import json
import random
from typing import List
from .base import BaseDataset, BaseSample
from .download import exist_or_download, local_files
from . import types


def load_locomo_dataset(max_samples: int = None) -> BaseDataset:
    exist_or_download("locomo")
    dataset_files = local_files("locomo")

    with open(dataset_files["locomo10.json"], "r") as f:
        data = json.load(f)

    return parse_locomo_data(data, max_samples)


def parse_locomo_data(data: dict, max_samples: int = None) -> BaseDataset:
    if max_samples is not None and max_samples > 0:
        data = data[:max_samples]

    samples = []
    for sample in data:
        conversations = parse_conversations(sample["conversation"])
        qas = parse_qas(sample)

        add_sample = BaseSample(
            sessions=conversations,
            qa_pairs=qas,
        )
        samples.append(add_sample)
    return BaseDataset(samples=samples)


def parse_session(
    session_data: List[dict], session_id: int, date_time: str
) -> types.Conversation:
    """Parse a single session's data, including turns with images by using their captions."""
    turns = []
    roles = ["user", "assistant"]
    for now_i, turn in enumerate(session_data):
        # For turns with images, combine caption and text
        text = turn.get("text", "")
        if "img_url" in turn and "blip_caption" in turn:
            caption_text = f"[Image: {turn['blip_caption']}]"
            if text:
                text = f"{caption_text} {text}"
            else:
                text = caption_text

        turns.append(
            types.Message(
                role=roles[now_i % 2],
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
    for key, value in conv_data.items():
        if key.startswith("session_") and isinstance(value, list):
            session_id = int(key.split("_")[1])
            date_time = conv_data.get(f"{key}_date_time")
            if date_time:
                session = parse_session(value, session_id, date_time)
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
                questions=[types.Message(role="user", content=use_prompt)],
                answer=str(use_answer),
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
    console.print(dataset.samples[0].sessions[0])
    console.print(dataset.samples[0].qa_pairs[0])

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str
    alias: str | None = None
    id: str | None = None
    date_string: str | None = None

    kwargs: dict | None = None


class Conversation(BaseModel):
    messages: list[Message]
    id: str | None = None
    date_string: str | None = None

    kwargs: dict | None = None


class QA(BaseModel):
    questions: list[Message]
    answer: str

    final_prompt: str

    kwargs: dict | None = None


class RemoteFile(BaseModel):
    url: str
    name: str
    hash: str

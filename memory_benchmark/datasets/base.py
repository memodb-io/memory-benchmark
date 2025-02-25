from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str
    alias: str | None = None
    id: str | None = None


class Conversation(BaseModel):
    messages: list[Message]
    id: str | None = None


class RemoteFile(BaseModel):
    url: str
    name: str
    hash: str

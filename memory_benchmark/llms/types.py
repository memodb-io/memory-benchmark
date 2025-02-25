from pydantic import BaseModel


class LLMResult(BaseModel):
    content: str
    input_tokens: int
    output_tokens: int

from openai import AsyncOpenAI
from ..env import CONFIG
from .types import LLMResult

_global_openai_async_client = None


def get_openai_async_client_instance() -> AsyncOpenAI:
    global _global_openai_async_client
    if _global_openai_async_client is None:
        _global_openai_async_client = AsyncOpenAI(
            base_url=CONFIG.llm_base_url,
            api_key=CONFIG.llm_api_key,
        )
    return _global_openai_async_client


async def openai_complete(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> LLMResult:
    openai_async_client = get_openai_async_client_instance()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, timeout=120, **kwargs
    )
    return LLMResult(
        content=response.choices[0].message.content,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
    )

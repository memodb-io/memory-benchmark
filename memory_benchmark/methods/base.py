from abc import ABC, abstractmethod
from ..datasets.types import Conversation, Message


class BaseMethod(ABC):
    @abstractmethod
    async def create_new_account(self, **kwargs) -> str:
        pass

    @abstractmethod
    async def cleanup_account(self, account_id: str, **kwargs):
        pass

    @abstractmethod
    async def insert_conversation(
        self, account_id: str, conversation: Conversation, **kwargs
    ):
        pass

    @abstractmethod
    async def get_memory(
        self,
        account_id: str,
        query_messages: list[Message],
        max_memory_token_size: int = None,
        **kwargs
    ) -> str:
        pass

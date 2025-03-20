from .base import BaseMethod
from ..datasets.types import Conversation, Message
from ..env import console

try:
    from memobase import MemoBaseClient, ChatBlob
except ImportError:
    MemoBaseClient = None


def pack_message(m: Message) -> dict:
    m_dict = {
        "role": m.role,
        "content": m.content,
    }
    if m.alias:
        m_dict["alias"] = m.alias
    if m.date_string:
        m_dict["created_at"] = m.date_string
    return m_dict


class MemoBase(BaseMethod):
    def __init__(
        self,
        memobase_api_key: str,
        memobase_project_url: str = "https://api.memobase.io",
    ):
        if MemoBaseClient is None:
            raise ImportError(
                "memobase is not installed, please setup this method first"
            )

        self.memobase_api_key = memobase_api_key
        self.memobase_project_url = memobase_project_url
        self.client = MemoBaseClient(
            api_key=memobase_api_key,
            project_url=memobase_project_url,
        )
        assert self.client.ping(), "Failed to connect to Memobase backend"

    async def create_new_account(self, **kwargs) -> str:
        uid = self.client.add_user()
        return uid

    async def cleanup_account(self, account_id: str, **kwargs):
        self.client.delete_user(account_id)

    async def insert_conversation(
        self, account_id: str, conversation: Conversation, **kwargs
    ):
        u = self.client.get_user(account_id, no_get=True)
        for i in range(0, len(conversation.messages), 2):
            messages = conversation.messages[i : i + 2]
            u.insert(ChatBlob(messages=[pack_message(m) for m in messages]))

    async def get_memory(
        self,
        account_id: str,
        query_messages: list[Message],
        max_memory_token_size: int,
        **kwargs,
    ) -> str:
        profiles = self.client.get_user(account_id).profile(
            max_token_size=max_memory_token_size,
            prefer_topics=["basic_info", "work", "interests"],
        )
        context = "- " + "\n- ".join(
            [f"{p.topic}::{p.sub_topic}: {p.content}" for p in profiles]
        )
        return context

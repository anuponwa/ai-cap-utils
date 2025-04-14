from typing import TypedDict
from langchain_core.messages.base import BaseMessage


class MessageDict(TypedDict):
    message: list[BaseMessage]

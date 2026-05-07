from pydantic import BaseModel
from typing import List, Tuple, Union


class ChatMessage(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]] = []


class SourceCitation(BaseModel):
    file: str
    page: Union[str, int]


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
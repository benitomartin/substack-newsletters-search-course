# src/models/qdrant_models.py
from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl


# -----------------------------
# Qdrant payload settings
# -----------------------------
class ArticleChunkPayload(BaseModel):
    feed_name: str = Field(default="", description="Name of the feed")
    feed_author: str = Field(default="", description="Author of the feed")
    article_authors: list[str] = Field(default_factory=list, description="Authors of the article")
    title: str = Field(default="", description="Title of the article")
    url: HttpUrl | str | None = Field(default=None, description="URL of the article")
    published_at: datetime | str = Field(
        default_factory=datetime.now, description="Publication date of the article"
    )
    created_at: datetime | str = Field(
        default_factory=datetime.now, description="Creation date of the article"
    )
    chunk_index: int = Field(default=0, description="Index of the article chunk")
    chunk_text: str | None = Field(default=None, description="Text content of the article chunk")

from pydantic import BaseModel, Field


# -----------------------------
# Feed settings
# -----------------------------
class FeedItem(BaseModel):
    name: str = Field(default="", description="Name of the feed")
    author: str = Field(default="", description="Author of the feed")
    url: str = Field(default="", description="URL of the feed")


# -----------------------------
# Article settings
# -----------------------------
class ArticleItem(BaseModel):
    feed_name: str = Field(default="", description="Name of the feed")
    feed_author: str = Field(default="", description="Author of the feed")
    title: str = Field(default="", description="Title of the article")
    url: str = Field(default="", description="URL of the article")
    content: str = Field(default="", description="Content of the article")
    article_authors: list[str] = Field(default_factory=list, description="Authors of the article")
    published_at: str | None = Field(default=None, description="Publication date of the article")
    # cover_image: str | None = None

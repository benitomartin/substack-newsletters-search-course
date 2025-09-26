import uuid
from uuid import UUID

from sqlalchemy import ARRAY, TIMESTAMP, BigInteger, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SubstackTestArticle(Base):
    __tablename__ = "substack_test"

    # Primary internal ID
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)

    # External unique identifier
    uuid: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        default=uuid.uuid4,
        unique=True,
        nullable=False,
        index=True,
    )

    # Article fields
    feed_name: Mapped[str] = mapped_column(String, nullable=False)
    feed_author: Mapped[str] = mapped_column(String, nullable=False)
    article_authors: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    url: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    published_at: Mapped[str] = mapped_column(TIMESTAMP, nullable=False)
    created_at: Mapped[str] = mapped_column(TIMESTAMP, server_default=func.now(), nullable=False)

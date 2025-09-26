from prefect import task
from prefect.cache_policies import NO_CACHE
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from src.config import settings
from src.infrastructure.supabase.init_session import init_session
from src.models.article_models import ArticleItem, FeedItem
from src.models.sql_models import SubstackArticle
from src.utils.logger_util import setup_logging


@task(
    task_run_name="batch_ingest-{feed.name}",
    description="Ingest already parsed RSS articles in batches.",
    retries=2,
    retry_delay_seconds=120,
    cache_policy=NO_CACHE,
)
def ingest_from_rss(
    fetched_articles: list[ArticleItem],
    feed: FeedItem,
    article_model: type[SubstackArticle],
    engine: Engine,
) -> None:
    """Ingest articles fetched from RSS (already Markdownified).

    Articles are inserted in batches to optimize database writes. Errors during
    ingestion of individual batches are logged but do not stop subsequent batches.

    Args:
        fetched_articles: List of ArticleItem objects to ingest.
        feed: The FeedItem representing the source feed.
        article_model: The SQLAlchemy model class for articles.
        engine: SQLAlchemy Engine for database connection.

    Raises:
        RuntimeError: If ingestion completes with errors.
    """

    logger = setup_logging()
    rss = settings.rss
    errors = []
    batch: list[ArticleItem] = []

    session: Session = init_session(engine)

    try:
        for i, article in enumerate(fetched_articles, start=1):
            batch.append(article)

            if len(batch) >= rss.batch_size:
                batch_num = i // rss.batch_size
                try:
                    _persist_batch(session, batch, article_model)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    logger.error(f"Failed to ingest batch {batch_num} for feed '{feed.name}': {e}")
                    errors.append(f"Batch {batch_num}")
                else:
                    logger.info(
                        f"🔁 Ingested batch {batch_num} with {len(batch)} articles "
                        f"for feed '{feed.name}'"
                    )
                batch = []

        # leftovers
        if batch:
            try:
                _persist_batch(session, batch, article_model)
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to ingest final batch for feed '{feed.name}': {e}")
                errors.append("Final batch")
            else:
                logger.info(
                    f"👉 Ingested final batch of {len(batch)} articles for feed '{feed.name}'"
                )

        if errors:
            raise RuntimeError(f"Ingestion completed with errors: {errors}")

    except Exception as e:
        logger.error(f"Unexpected error in ingest_from_rss for feed '{feed.name}': {e}")
        raise
    finally:
        session.close()
        logger.info(f"Database session closed for feed '{feed.name}'")


def _persist_batch(
    session: Session,
    batch: list[ArticleItem],
    article_model: type[SubstackArticle],
) -> None:
    """Helper to bulk insert a batch of ArticleItems."""
    rows = [
        article_model(
            feed_name=article.feed_name,
            feed_author=article.feed_author,
            title=article.title,
            url=article.url,
            content=article.content,
            article_authors=article.article_authors,
            published_at=article.published_at,
        )
        for article in batch
    ]
    session.bulk_save_objects(rows)

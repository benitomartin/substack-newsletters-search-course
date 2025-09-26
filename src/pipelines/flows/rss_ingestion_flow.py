from prefect import flow, unmapped

from src.config import settings
from src.infrastructure.supabase.init_session import init_engine
from src.models.article_models import FeedItem
from src.models.sql_models import SubstackArticle
from src.pipelines.tasks.fetch_rss import fetch_rss_entries
from src.pipelines.tasks.ingest_rss import ingest_from_rss
from src.utils.logger_util import setup_logging


@flow(
    name="rss_ingest_flow",
    flow_run_name="rss_ingest_flow_run",
    description="Fetch and ingest articles from RSS feeds.",
    retries=2,
    retry_delay_seconds=120,
)
def rss_ingest_flow(article_model: type[SubstackArticle] = SubstackArticle) -> None:
    """Fetch and ingest articles from configured RSS feeds concurrently.

    Each feed is fetched in parallel and ingested into the database
    with error handling at each stage. Ensures the database engine is disposed
    after completion.

    Args:
        article_model (type[SubstackArticle]): SQLAlchemy model for storing articles.

    Returns:
        None

    Raises:
        RuntimeError: If ingestion fails for all feeds.
        Exception: For unexpected errors during execution.
    """
    logger = setup_logging()
    engine = init_engine()
    errors = []

    # tracking counters
    per_feed_counts: dict[str, int] = {}
    total_ingested = 0

    try:
        if not settings.rss.feeds:
            logger.warning("No feeds found in configuration.")
            return

        feeds = [FeedItem(name=f.name, author=f.author, url=f.url) for f in settings.rss.feeds]
        logger.info(f"🚀 Processing {len(feeds)} feeds concurrently...")

        # 1. Fetch articles concurrently
        fetched_articles_futures = fetch_rss_entries.map(
            feeds,
            engine=unmapped(engine),
            article_model=unmapped(article_model),
        )

        # 2. Ingest concurrently per feed
        results = []
        for feed, fetched_future in zip(feeds, fetched_articles_futures, strict=False):
            try:
                fetched_articles = fetched_future.result()
            except Exception as e:
                logger.error(f"❌ Error fetching articles for feed '{feed.name}': {e}")
                errors.append(f"Fetch error: {feed.name}")
                continue

            if not fetched_articles:
                logger.info(f"📭 No new articles for feed '{feed.name}'")
                per_feed_counts[feed.name] = 0
                continue

            try:
                count = len(fetched_articles)
                per_feed_counts[feed.name] = count
                total_ingested += count
                logger.info(f"✅ Feed '{feed.name}': {count} articles ready for ingestion")

                task_result = ingest_from_rss.submit(
                    fetched_articles,
                    feed,
                    article_model=article_model,
                    engine=engine,
                )
                results.append(task_result)
            except Exception as e:
                logger.error(f"❌ Error submitting ingest_from_rss for feed '{feed.name}': {e}")
                errors.append(f"Ingest error: {feed.name}")

        # 3. Wait for all ingestion tasks
        for r in results:
            try:
                r.result()
            except Exception as e:
                logger.error(f"❌ Error in ingest_from_rss task: {e}")
                errors.append("Task failure")

        # ---- Summary logging ----
        logger.info("📊 Ingestion Summary per feed:")
        for feed_name, count in per_feed_counts.items():
            logger.info(f"   • {feed_name}: {count} article(s) ingested")

        logger.info(f"📝 Total ingested across all feeds: {total_ingested}")

        if errors:
            raise RuntimeError(f"Flow completed with errors: {errors}")

    except Exception as e:
        logger.error(f"💥 Unexpected error in rss_ingest_flow: {e}")
        raise
    finally:
        engine.dispose()
        logger.info("🔒 Database engine disposed.")


if __name__ == "__main__":
    rss_ingest_flow(article_model=SubstackArticle)

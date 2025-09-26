import pytest
import responses
from loguru import logger
from sqlalchemy import text
from sqlalchemy.orm import Session
from test_models.test_sql_models import SubstackTestArticle

from src.infrastructure.supabase.init_session import init_engine
from src.models.article_models import ArticleItem, FeedItem
from src.pipelines.tasks.fetch_rss import fetch_rss_entries


@pytest.mark.unit
@responses.activate
def test_fetch_rss_mocked_feed() -> None:
    """Unit test that fetches a mocked RSS feed instead of hitting the real URL,
    ensuring the test DB is empty beforehand.
    """
    test_feed = FeedItem(
        name="Test Feed",
        author="Unit Test Author",
        url="https://decodingml.substack.com/feed",
    )

    # Mock a minimal RSS response
    responses.add(
        responses.GET,
        test_feed.url,
        body="""
        <rss version="2.0">
          <channel>
            <title>Test Feed</title>
            <item>
              <title>Test Article</title>
              <link>https://example.com/test-article</link>
              <description>Test description</description>
              <pubDate>Mon, 01 Jan 2025 00:00:00 +0000</pubDate>
            </item>
          </channel>
        </rss>
        """,
        status=200,
        content_type="application/rss+xml",
    )

    engine = init_engine()
    session: Session | None = None

    try:
        # Clear the test table before running
        session = Session(bind=engine)
        logger.info("Clearing test table 'substack_test' before test")
        session.execute(text("DELETE FROM substack_test"))
        session.commit()
        logger.info("Test table cleared")

        # Fetch articles from mocked feed
        articles = fetch_rss_entries(
            feed=test_feed,
            engine=engine,
            article_model=SubstackTestArticle,
        )
        logger.info(f"Fetched {len(articles)} articles from {test_feed.url}")

        # Assertions
        assert isinstance(articles, list)
        assert all(isinstance(a, ArticleItem) for a in articles)
        assert len(articles) > 0, "No articles were fetched"
        assert articles[0].title == "Test Article"

    finally:
        if session:
            session.close()
            logger.info("Test database session closed")
        engine.dispose()
        logger.info("SQLAlchemy engine disposed after test")

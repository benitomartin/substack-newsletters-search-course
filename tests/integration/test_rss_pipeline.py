import pytest
import responses
from loguru import logger
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from test_models.test_sql_models import SubstackTestArticle  # Test-specific table model

from src.models.article_models import FeedItem
from src.pipelines.tasks.fetch_rss import fetch_rss_entries
from src.pipelines.tasks.ingest_rss import ingest_from_rss


@pytest.mark.integration
@responses.activate
def test_rss_pipeline_end_to_end_mocked(db_session: Session, db_engine: Engine) -> None:
    """Integration test for the RSS pipeline using mocked HTTP requests.

    This avoids hitting live RSS feeds or article URLs, making the test CI-safe.
    1. Clears the test table.
    2. Mocks fetching articles from an RSS feed.
    3. Mocks parsing article content.
    4. Ingests articles into the test table.
    5. Verifies insertion and basic correctness.

    Args:
        db_session (Session): SQLAlchemy session for DB interactions.
        db_engine (Engine): SQLAlchemy engine for task-level operations.
    """

    # Clear test table
    logger.info("Clearing test table 'substack_test'")
    db_session.execute(text("DELETE FROM substack_test"))
    db_session.commit()

    # Verify table is empty
    initial_count = db_session.query(SubstackTestArticle).count()
    logger.info(f"Initial article count in test table: {initial_count}")
    assert initial_count == 0, "Test table was not cleared"

    # Mock RSS feed URL
    feed_url = "https://aiechoes.substack.com/feed"
    responses.add(
        responses.GET,
        feed_url,
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

    # Mock the article page with the div your parser expects
    responses.add(
        responses.GET,
        "https://example.com/test-article",
        body="""
        <html>
          <body>
            <div class="post-body">
              <p>This is the article content</p>
            </div>
          </body>
        </html>
        """,
        status=200,
        content_type="text/html",
    )

    # Define test feed
    test_feed = FeedItem(
        name="Test Feed",
        author="Test Author",
        url=feed_url,
    )

    # Fetch articles (mocked feed)
    fetched_articles = fetch_rss_entries(
        test_feed,
        engine=db_engine,
        article_model=SubstackTestArticle,
    )
    logger.info(f"Fetched {len(fetched_articles)} articles for feed '{test_feed.name}'")

    # Ensure we have articles
    assert fetched_articles, "No articles were fetched from mocked feed"

    # Ingest parsed articles
    ingest_from_rss(
        fetched_articles,
        feed=test_feed,
        article_model=SubstackTestArticle,
        engine=db_engine,
    )

    # Verify DB insertion
    articles_in_db = (
        db_session.query(SubstackTestArticle)
        .order_by(SubstackTestArticle.published_at.desc())
        .all()
    )
    logger.info(f"Inserted article titles: {[a.title for a in articles_in_db]}")
    assert articles_in_db, "No articles were inserted into the test table"

    # Check at least the first fetched article was inserted
    first_fetched_title = fetched_articles[0].title
    titles_in_db = [a.title for a in articles_in_db]
    assert first_fetched_title in titles_in_db, (
        f"First fetched article '{first_fetched_title}' not found in DB"
    )


################################################################################
# The code below calls out to live URLs and is not suitable for CI,
# as Substack may block requests from CI environments.
# It is left here for reference and can be run manually if desired.
# Uncomment to enable live integration test


# import pytest
# from loguru import logger
# from sqlalchemy import text
# from sqlalchemy.engine import Engine
# from sqlalchemy.orm import Session
# from test_models.test_sql_models import SubstackTestArticle  # Test-specific table model

# from src.models.article_models import FeedItem
# from src.pipelines.tasks.batch_parse_ingest_articles import parse_and_ingest
# from src.pipelines.tasks.fetch_rss import fetch_rss_entries


# @pytest.mark.integration
# def test_rss_pipeline_end_to_end(db_session: Session, db_engine: Engine) -> None:
#     """Integration test for the end-to-end RSS pipeline:
#     1. Clears the test table.
#     2. Fetches articles from a live RSS feed.
#     3. Parses and ingests articles into the test table.
#     4. Verifies insertion and basic correctness.

#     Args:
#         db_session (Session): SQLAlchemy session for DB interactions.
#         db_engine (Engine): SQLAlchemy engine for task-level operations.

#     """
#     # Clear test table
#     logger.info("Clearing test table 'substack_test'")
#     db_session.execute(text("DELETE FROM substack_test"))
#     db_session.commit()

#     # Verify table is empty
#     initial_count = db_session.query(SubstackTestArticle).count()
#     logger.info(f"Initial article count in test table: {initial_count}")
#     assert initial_count == 0, "Test table was not cleared"

#     # Define test feed
#     test_feed = FeedItem(
#         name="Test Feed",
#         author="Test Author",
#         url="https://aiechoes.substack.com/feed",
#     )

#     # Fetch articles
#     fetched_articles = fetch_rss_entries(
#         test_feed,
#         engine=db_engine,
#         article_model=SubstackTestArticle,
#     )
#     logger.info(f"Fetched {len(fetched_articles)} articles for feed '{test_feed.name}'")

#     if not fetched_articles:
#         logger.warning("No articles fetched; skipping test due to empty RSS feed")
#         pytest.skip("No new articles available in the RSS feed")

#     # Parse and ingest
#     parse_and_ingest(
#         fetched_articles,
#         feed=test_feed,
#         article_model=SubstackTestArticle,
#         engine=db_engine,
#     )

#     # Verify DB insertion
#     articles_in_db = (
#         db_session.query(SubstackTestArticle)
#         .order_by(SubstackTestArticle.published_at.desc())
#         .all()
#     )
#     logger.info(f"Inserted article titles: {[a.title for a in articles_in_db]}")
#     assert articles_in_db, "No articles were inserted into the test table"

#     # Check at least the first fetched article was inserted
#     first_fetched_title = fetched_articles[0].title
#     titles_in_db = [a.title for a in articles_in_db]
#     assert first_fetched_title in titles_in_db, (
#         f"First fetched article '{first_fetched_title}' not found in DB"
#     )

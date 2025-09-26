import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from prefect import task
from prefect.cache_policies import NO_CACHE
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from src.infrastructure.supabase.init_session import init_session
from src.models.article_models import ArticleItem, FeedItem
from src.models.sql_models import SubstackArticle
from src.utils.logger_util import setup_logging


@task(
    task_run_name="fetch_rss_entries-{feed.name}",
    description="Fetch RSS entries from a Substack feed.",
    retries=2,
    retry_delay_seconds=120,
    cache_policy=NO_CACHE,
)
def fetch_rss_entries(
    feed: FeedItem,
    engine: Engine,
    article_model: type[SubstackArticle] = SubstackArticle,
) -> list[ArticleItem]:
    """Fetch all RSS items from a Substack feed and convert them to ArticleItem objects.

    Each task uses its own SQLAlchemy session. Articles already stored in the database
    or with empty links/content are skipped. Errors during parsing individual items
    are logged but do not stop processing.

    Args:
        feed (FeedItem): Metadata for the feed (name, author, URL).
        engine (Engine): SQLAlchemy engine for database connection.
        article_model (type[SubstackArticle], optional): Model used to persist articles.
            Defaults to SubstackArticle.

    Returns:
        list[ArticleItem]: List of new ArticleItem objects ready for parsing/ingestion.

    Raises:
        RuntimeError: If the RSS fetch fails.
        Exception: For unexpected errors during execution.
    """

    logger = setup_logging()
    session: Session = init_session(engine)
    items: list[ArticleItem] = []

    try:
        try:
            response = requests.get(feed.url, timeout=15)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch feed '{feed.name}': {e}")
            raise RuntimeError(f"RSS fetch failed for feed '{feed.name}'") from e

        soup = BeautifulSoup(response.content, "xml")
        rss_items = soup.find_all("item")

        for _, item in enumerate(rss_items):
            try:
                link = item.find("link").get_text(strip=True) if item.find("link") else ""  # type: ignore
                if not link or session.query(article_model).filter_by(url=link).first():
                    logger.info(
                        f"Skipping already stored or empty-link article for feed '{feed.name}'"
                    )
                    continue

                title = (
                    item.find("title").get_text(strip=True) if item.find("title") else "Untitled"  # type: ignore
                )

                # Prefer full text in <content:encoded>
                content_elem = item.find("content:encoded") or item.find("description")  # type: ignore
                raw_html = content_elem.get_text() if content_elem else ""
                content_md = ""

                # 🚨 Skip if article contains a self-referencing "Read more" link
                if raw_html:
                    try:
                        html_soup = BeautifulSoup(raw_html, "html.parser")
                        for a in html_soup.find_all("a", href=True):
                            if (
                                a["href"].strip() == link  # type: ignore
                                and "read more" in a.get_text(strip=True).lower()
                            ):
                                logger.info(f"Paywalled/truncated article skipped: '{title}'")
                                raise StopIteration  # skip this item
                    except StopIteration:
                        continue
                    except Exception as e:
                        logger.warning(f"Failed to inspect links for '{title}': {e}")

                if raw_html:
                    try:
                        content_md = md(
                            raw_html,
                            strip=["script", "style"],
                            heading_style="ATX",
                            bullets="*",
                            autolinks=True,
                        )
                        content_md = "\n".join(
                            line.strip() for line in content_md.splitlines() if line.strip()
                        )
                    except Exception as e:
                        logger.warning(f"Markdown conversion failed for '{title}': {e}")
                        content_md = raw_html

                if not content_md:
                    logger.warning(f"Skipping article '{title}' with empty content")
                    continue

                author_elem = item.find("creator") or item.find("dc:creator")  # type: ignore
                author = author_elem.get_text(strip=True) if author_elem else feed.author

                pub_date_elem = item.find("pubDate")  # type: ignore
                pub_date_str = pub_date_elem.get_text(strip=True) if pub_date_elem else None

                article_item = ArticleItem(
                    feed_name=feed.name,
                    feed_author=feed.author,
                    title=title,
                    url=link,
                    content=content_md,
                    article_authors=[author] if author else [],
                    published_at=pub_date_str,
                )
                items.append(article_item)

            except Exception as e:
                logger.error(f"Error processing RSS item for feed '{feed.name}': {e}")
                continue

        logger.info(f"Fetched {len(items)} new articles for feed '{feed.name}'")
        return items

    except Exception as e:
        logger.error(f"Unexpected error in fetch_rss_entries for feed '{feed.name}': {e}")
        raise
    finally:
        session.close()
        logger.info(f"Database session closed for feed '{feed.name}'")


# if __name__ == "__main__":
#     from src.infrastructure.supabase.init_session import init_engine

#     engine = init_engine()
#     test_feed = FeedItem(
#         name="AI Echoes",
#         author="Benito Martin",
#         url="https://aiechoes.substack.com/feed"
#     )
#     articles = fetch_rss_entries(test_feed, engine)
#     print(f"Fetched {len(articles)} articles.")

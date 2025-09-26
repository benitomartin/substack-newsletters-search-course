import asyncio

from src.infrastructure.qdrant.qdrant_vectorstore import AsyncQdrantVectorStore
from src.utils.logger_util import setup_logging

logger = setup_logging()


async def main() -> None:
    """Create necessary indexes for the Qdrant vector store.

    Initializes an AsyncQdrantVectorStore and creates HNSW, title, article authors,
    feed author, and feed name indexes. Logs errors and ensures proper execution.

    Args:
        None

    Returns:
        None

    Raises:
        RuntimeError: If an error occurs during index creation.
        Exception: For unexpected errors during execution.

    """
    logger.info("Creating Qdrant indexes")
    try:
        vectorstore = AsyncQdrantVectorStore()
        await vectorstore.enable_hnsw()
        await vectorstore.create_title_index()
        await vectorstore.create_article_authors_index()
        await vectorstore.create_feed_author_index()
        await vectorstore.create_article_feed_name_index()
        logger.info("Qdrant indexes created successfully")
    except RuntimeError as e:
        logger.error(f"Failed to create Qdrant indexes: {e}")
        raise RuntimeError("Error creating Qdrant indexes") from e
    except Exception as e:
        logger.error(f"Unexpected error creating Qdrant indexes: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

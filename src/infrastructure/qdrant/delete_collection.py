import asyncio

from src.infrastructure.qdrant.qdrant_vectorstore import AsyncQdrantVectorStore
from src.utils.logger_util import setup_logging

logger = setup_logging()


async def main() -> None:
    """Delete the Qdrant collection.

    Initializes an AsyncQdrantVectorStore and deletes its associated collection.
    Logs errors and ensures proper execution.

    Args:
        None

    Returns:
        None

    Raises:
        RuntimeError: If an error occurs during collection deletion.
        Exception: For unexpected errors during execution.

    """
    logger.info("Deleting Qdrant collection")
    try:
        vectorstore = AsyncQdrantVectorStore()
        await vectorstore.delete_collection()
        logger.info("Qdrant collection deleted successfully")
    except RuntimeError as e:
        logger.error(f"Failed to delete Qdrant collection: {e}")
        raise RuntimeError("Error deleting Qdrant collection") from e
    except Exception as e:
        logger.error(f"Unexpected error deleting Qdrant collection: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

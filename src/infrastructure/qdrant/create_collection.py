import asyncio

from src.infrastructure.qdrant.qdrant_vectorstore import AsyncQdrantVectorStore
from src.utils.logger_util import setup_logging

logger = setup_logging()


async def main() -> None:
    """Create a Qdrant collection asynchronously using AsyncQdrantVectorStore.

    This function initializes an AsyncQdrantVectorStore instance and calls its
    create_collection method to set up a Qdrant collection for vector storage.
    Errors during collection creation are logged
    and handled gracefully.

    Args:
        None

    Returns:
        None

    Raises:
        RuntimeError: If an error occurs during Qdrant collection creation.
        Exception: For unexpected errors during execution.

    """
    # Initialize the logger
    logger.info("Creating Qdrant collection")

    try:
        # Initialize the AsyncQdrantVectorStore instance
        vectorstore = AsyncQdrantVectorStore()
        # Create the Qdrant collection asynchronously
        await vectorstore.create_collection()
        logger.info("Qdrant collection created successfully")

    except RuntimeError as e:
        logger.error(f"Failed to create Qdrant collection: {e}")
        raise RuntimeError("Error creating Qdrant collection") from e
    except Exception as e:
        logger.error(f"Unexpected error during Qdrant collection creation: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

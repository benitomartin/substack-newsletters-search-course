import gc
import os
from datetime import datetime

import dotenv
from prefect import task

from src.infrastructure.qdrant.qdrant_vectorstore import AsyncQdrantVectorStore
from src.infrastructure.supabase.init_session import init_engine, init_session
from src.utils.logger_util import setup_logging

dotenv.load_dotenv()


@task(
    task_run_name="ingest_qdrant",
    description="Ingest articles from SQL to Qdrant.",
    retries=2,
    retry_delay_seconds=120,
)
async def ingest_qdrant(from_date: datetime | None = None):
    """Ingest articles from SQL database into Qdrant vector store.

    Args:
        from_date (datetime | None, optional): Only ingest articles published after this date.
            Defaults to None (ingest all articles).

    Returns:
        None

    Raises:
        RuntimeError: If ingestion fails.
        Exception: For unexpected errors during execution.

    """
    logger = setup_logging()
    logger.info(f"Starting Qdrant ingestion task from_date={from_date}")

    logger.info(f"QDRANT_URL: {os.getenv('QDRANT__URL')}")

    vectorstore = AsyncQdrantVectorStore()
    engine = init_engine()
    session = init_session(engine)

    try:
        await vectorstore.ingest_from_sql(session=session, from_date=from_date)
    except Exception as e:
        logger.error(f"Unexpected error during Qdrant ingestion: {e}")
        raise RuntimeError("Qdrant ingestion failed") from e
    finally:
        # Cleanup resources
        session.close()
        await vectorstore.client.close()
        gc.collect()
        logger.info("Qdrant ingestion task complete and resources cleaned up")

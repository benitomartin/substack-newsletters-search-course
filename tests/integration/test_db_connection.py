from loguru import logger
from sqlalchemy import text
from sqlalchemy.engine import Connection

from src.utils.logger_util import setup_logging

setup_logging()


def test_connect_to_test_table(db_session: Connection) -> None:
    """Test connectivity to the 'substack_test' table and fetch a single row.

    Args:
        db_session (Connection): SQLAlchemy Connection object.

    Raises:
        AssertionError: If the query result is not a list.
        Exception: If the table does not exist or query fails.

    """
    logger.info("Testing connection to 'substack_test' table...")

    try:
        result = db_session.execute(text("SELECT * FROM substack_test LIMIT 1")).fetchall()
        logger.info(f"Query result: {result}")
        assert isinstance(result, list), "Query result is not a list"
    except Exception as e:
        logger.error(f"Failed to query 'substack_test' table: {e}")
        raise

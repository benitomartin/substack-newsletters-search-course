from sqlalchemy import inspect
from sqlalchemy.exc import SQLAlchemyError

from src.infrastructure.supabase.init_session import init_engine
from src.models.sql_models import Base
from src.utils.logger_util import setup_logging

logger = setup_logging()


def delete_all_tables() -> None:
    """Drop all tables defined in the SQLAlchemy Base metadata from the Supabase Postgres database.

    This function initializes a SQLAlchemy engine, checks for existing tables, and drops them
    after user confirmation to prevent accidental data loss. It is a destructive operation and
    should be used with caution. The engine is disposed of after the operation to release resources.
    Errors during table deletion are logged and handled gracefully.

    Args:
        None

    Returns:
        None

    Raises:
        SQLAlchemyError: If an error occurs during database operations (e.g., connection issues).
        Exception: For unexpected errors during table inspection or deletion.

    """
    # Initialize the SQLAlchemy engine
    engine = init_engine()
    try:
        # Create an inspector to check existing tables
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()

        # Check if there are any tables to delete
        if not existing_tables:
            logger.info("No tables found in the database. Nothing to delete.")
            return

        # Prompt user for confirmation to prevent accidental data loss
        confirm = input(
            f"Are you sure you want to DROP ALL tables? {existing_tables}\n"
            "Type 'YES' to confirm or any other key to cancel: "
        )
        if confirm != "YES":
            logger.info("Operation canceled by user.")
            return

        # Drop all tables defined in Base.metadata
        logger.info(f"Dropping all tables: {existing_tables}")
        Base.metadata.drop_all(bind=engine)
        logger.info("All tables dropped successfully.")

    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error dropping tables: {e}")
        raise SQLAlchemyError("Failed to drop tables from the database") from e
    except Exception as e:
        logger.error(f"Unexpected error dropping tables: {e}")
        raise
    finally:
        # Dispose of the engine to release connections
        engine.dispose()
        logger.info("Database engine disposed.")


if __name__ == "__main__":
    delete_all_tables()

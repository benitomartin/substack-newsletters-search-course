from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from src.config import settings
from src.utils.logger_util import setup_logging

logger = setup_logging()


def init_engine() -> Engine:
    """Initialize the SQLAlchemy engine for Supabase Postgres.

    Returns:
        Engine: The SQLAlchemy engine instance.

    Raises:
        ValueError: If database configuration is missing or invalid.
        OperationalError: If the database connection fails.
        SQLAlchemyError: For other SQLAlchemy-related errors during engine creation.

    """
    try:
        db = settings.supabase_db
        if not all([db.user, db.password, db.host, db.port, db.name]):
            logger.error(
                "Incomplete database configuration: missing user, password, host, port, or name"
            )
            raise ValueError(
                "Incomplete database configuration: ensure all Supabase settings are provided"
            )

        logger.info(f"Connecting to database {db.name} at {db.host}:{db.port}")
        engine_url = (
            f"postgresql://{db.user}:{db.password.get_secret_value()}@{db.host}:{db.port}/{db.name}"
        )
        logger.debug(f"Using engine URL: {engine_url}")

        # Create the engine with connection pooling options for robustness
        engine = create_engine(
            engine_url,
            pool_size=5,  # Matches number of feeds/tasks
            max_overflow=10,  # Allow additional connections if pool is full
            pool_timeout=30,  # Timeout for getting a connection from the pool
            echo=False,  # Disable SQL statement logging (set to True for debugging)
            connect_args={
                "client_encoding": "utf8",
            },
        )

        # Test the connection to ensure it’s valid
        with engine.connect():
            logger.debug("Successfully tested database connection")

        logger.info("Database engine initialized successfully")
        return engine

    except AttributeError as e:
        logger.error(f"Invalid database configuration: {e}")
        raise ValueError(
            "Invalid database configuration: ensure settings.supabase_db is properly configured"
        ) from e
    except OperationalError as e:
        logger.error(f"Failed to connect to database: {e}")
        raise
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error during engine initialization: {e}")
        raise SQLAlchemyError("Failed to initialize database engine") from e
    except Exception as e:
        logger.error(f"Unexpected error during engine initialization: {e}")
        raise


def init_session(engine: Engine | None = None) -> Session:
    """Create a new SQLAlchemy session.

    Args:
        engine (Optional[Engine]): The SQLAlchemy engine to bind the session to.
        If None, a new engine is created.

    Returns:
        Session: A new SQLAlchemy session.

    Raises:
        ValueError: If no engine is provided and a new engine cannot be created.
        SQLAlchemyError: If session creation fails.

    """
    try:
        if engine is None:
            logger.debug("No engine provided; creating a new engine")
            engine = init_engine()

        logger.info("Creating new database session")
        SessionLocal = sessionmaker(
            bind=engine,
            autocommit=False,
            autoflush=False,
        )
        session = SessionLocal()
        logger.info("Database session created successfully")
        return session

    except ValueError as e:
        logger.error(f"Failed to create session due to invalid engine: {e}")
        raise ValueError("Cannot create session: invalid or missing engine") from e
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error during session creation: {e}")
        raise SQLAlchemyError("Failed to create database session") from e
    except Exception as e:
        logger.error(f"Unexpected error during session creation: {e}")
        raise

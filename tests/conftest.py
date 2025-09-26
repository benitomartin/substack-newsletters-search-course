import shutil
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session as SQLAlchemySession
from sqlalchemy.orm import sessionmaker

# Add src to Python path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import settings
from utils.logger_util import setup_logging

logger = setup_logging()
db = settings.supabase_db

DATABASE_URL = (
    f"postgresql://{db.user}:{db.password.get_secret_value()}@{db.host}:{db.port}/{db.name}"
)


@pytest.fixture(scope="session")
def db_engine() -> Generator[Engine, None, None]:
    """Create a SQLAlchemy engine for the test database session.
    Disposes the engine after the test session completes.

    Args:
        None
    Yields:
        Engine: A SQLAlchemy engine connected to the test database.
    """
    logger.info("Creating test database engine")
    engine = create_engine(DATABASE_URL)
    yield engine
    logger.info("Disposing test database engine")
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine: Engine) -> Generator[SQLAlchemySession, None, None]:
    """Provide a SQLAlchemy session for a single test function.
    Closes the session after the test finishes.

    Args:
        db_engine (Engine): The SQLAlchemy engine to bind the session to.
    Yields:
        SQLAlchemySession: A SQLAlchemy session connected to the test database.
    """
    logger.info("Creating test database session")
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()
    logger.info("Closed test database session")


@pytest.fixture(scope="function", autouse=True)
def clear_prefect_cache() -> Generator[None, None, None]:
    """Automatically clear Prefect cache before and after each test function
    to prevent interference between tests.

    Args:
        None
    Yields:
        None
    """
    prefect_dir = Path(".prefect")
    logger.debug("Clearing Prefect cache before test")
    if prefect_dir.exists():
        shutil.rmtree(prefect_dir, ignore_errors=True)
    yield
    if prefect_dir.exists():
        shutil.rmtree(prefect_dir, ignore_errors=True)
    logger.debug("Cleared Prefect cache after test")

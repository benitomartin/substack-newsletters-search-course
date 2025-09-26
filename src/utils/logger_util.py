import os
import sys
from typing import Any

import psutil
from loguru import logger as loguru_logger
from prefect.context import get_run_context
from prefect.logging import get_run_logger


def setup_logging(log_level: str | None = None):
    """Returns a logger configured for the current environment.

    - Inside Prefect flow/task: Prefect's run logger (`logging.Logger`).
    - Outside Prefect: Loguru logger.

    Args:
        log_level (str | None): Logging level to use (DEBUG, INFO, WARNING, ERROR).
                                 Defaults to LOG_LEVEL env variable or DEBUG.

    Returns:
        logging.Logger | loguru.Logger: Configured logger instance.

    """
    log_level = log_level or os.getenv("LOG_LEVEL", "DEBUG").upper()

    try:
        # Inside Prefect
        get_run_context()
        logger = get_run_logger()
        logger.setLevel(log_level)
        logger.debug(f"Logging initialized at {log_level} level (Prefect).")
        return logger
    except RuntimeError:
        # Outside Prefect → Loguru
        loguru_logger.remove()
        loguru_logger.add(
            sys.stdout,
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan> - "
            "<level>{message}</level>",
        )
        loguru_logger.debug(f"Logging initialized at {log_level} level (Loguru).")
        return loguru_logger


def log_batch_status(
    logger: Any,
    action: str,
    batch_size: int,
    total_articles: int | None = None,
    total_chunks: int | None = None,
    context: str = "",
) -> str:
    """Log batch action details along with current process and system memory usage.

    Args:
        logger (Any): Logger instance to use (Prefect or Loguru).
        action (str): Action description (e.g., 'Ingested', 'Parsed').
        batch_size (int): Number of items in the batch.
        total_articles (int | None): Total articles processed so far.
        total_chunks (int | None): Total chunks processed so far.
        context (str, optional): Additional context info.

    Returns:
        str: Formatted log string (useful for testing).

    """
    process = psutil.Process()
    mem = process.memory_info()
    rss_mb = mem.rss / 1024 / 1024
    vms_mb = mem.vms / 1024 / 1024

    svmem = psutil.virtual_memory()
    sys_used_mb = svmem.used / 1024 / 1024
    sys_percent = svmem.percent

    details = (
        f"{action} | batch_size={batch_size}"
        f"{f', total_articles={total_articles}' if total_articles is not None else ''}"
        f"{f', total_chunks={total_chunks}' if total_chunks is not None else ''}"
        f"{f', context={context}' if context else ''}"
        f" | process_mem: RSS={rss_mb:.1f}MB, VMS={vms_mb:.1f}MB"
        f" | system_mem: used={sys_used_mb:.1f}MB ({sys_percent:.0f}%)"
    )
    logger.info(details)
    return details

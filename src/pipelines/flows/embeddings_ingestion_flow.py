import argparse
import asyncio
from datetime import UTC, datetime, timedelta

from dateutil import parser
from prefect import flow, get_client
from prefect.client.schemas.filters import FlowFilter, FlowRunFilter
from prefect.client.schemas.sorting import FlowRunSort

from src.config import settings
from src.pipelines.tasks.ingest_embeddings import ingest_qdrant
from src.utils.logger_util import setup_logging


async def get_last_successful_run(flow_name: str) -> datetime | None:
    """Get the start time of the last successfully completed run for a given flow.

    Queries the Prefect API for recent completed runs of the exact flow `flow_name`.
    Returns the start time of the most recent completed run, or None if no runs exist.

    Args:
        flow_name (str): Exact name of the Prefect flow.

    Returns:
        datetime | None: Start time of the last completed run, or None if no run exists.

    Raises:
        Exception: If Prefect API calls fail unexpectedly.

    """
    logger = setup_logging()
    logger.info(f"Looking for last successful run of flow: {flow_name}")

    try:
        async with get_client() as client:
            # Step 1: get flows matching the name
            flows = await client.read_flows(flow_filter=FlowFilter(name=dict(eq_=flow_name)))  # type: ignore
            logger.debug(f"Flows returned by Prefect API: {flows}")

            exact_flow = next((f for f in flows if f.name == flow_name), None)
            if not exact_flow:
                logger.info(f"No flow found with exact name: {flow_name}")
                return None

            logger.info(f"Exact flow found: {exact_flow.id} ({exact_flow.name})")

            # Step 2: get recent completed runs
            flow_runs = await client.read_flow_runs(
                flow_run_filter=FlowRunFilter(state=dict(type=dict(any_=["COMPLETED"]))),  # type: ignore
                sort=FlowRunSort.START_TIME_DESC,
                limit=10,
            )
            logger.debug(f"Recent completed runs fetched: {[r.id for r in flow_runs]}")

            # Step 3: ensure only runs for this flow
            flow_runs = [r for r in flow_runs if r.flow_id == exact_flow.id]
            logger.debug(f"Filtered runs for exact flow: {[r.id for r in flow_runs]}")

            if not flow_runs:
                logger.info(f"No completed runs found for flow: {flow_name}")
                return None

            last_run_time = flow_runs[0].start_time
            logger.info(f"Last completed run for flow '{flow_name}' started at {last_run_time}")

            return last_run_time

    except Exception as e:
        logger.error(f"Error fetching last successful run for flow '{flow_name}': {e}")
        raise


@flow(
    name="qdrant_ingest_flow",
    flow_run_name="qdrant_ingest_flow_run",
    description="Orchestrates SQL → Qdrant ingestion",
    retries=2,
    retry_delay_seconds=120,
)
async def qdrant_ingest_flow(from_date: str | None = None) -> None:
    """Prefect Flow: Orchestrates ingestion of articles from SQL into Qdrant.

    Determines the starting cutoff date for ingestion (user-provided, last run date,
    or default fallback) and runs the Qdrant ingestion task.

    Args:
        from_date (str | None, optional): Start date in YYYY-MM-DD format. If None,
            falls back to last successful run or the configured default.

    Returns:
        None

    Raises:
        RuntimeError: If ingestion fails.
        Exception: For unexpected errors during execution.

    """
    logger = setup_logging()
    rss = settings.rss

    try:
        if from_date:
            # Parse user-provided date and assume UTC midnight
            from_date_dt = parser.parse(from_date).replace(tzinfo=UTC)
            logger.info(f"Using user-provided from_date: {from_date_dt}")
        else:
            # Fallback to last run date, default_start_date, or 30 days ago
            last_run_date = await get_last_successful_run("qdrant_ingest_flow")
            from_date_dt = (
                last_run_date
                or parser.parse(rss.default_start_date).replace(tzinfo=UTC)
                or (datetime.now(UTC) - timedelta(days=30))
            )
            logger.info(f"Using fallback from_date: {from_date_dt}")

        await ingest_qdrant(from_date=from_date_dt)

    except Exception as e:
        logger.error(f"Error during Qdrant ingestion flow: {e}")
        raise RuntimeError("Qdrant ingestion flow failed") from e


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--from-date",
        type=str,
        default=None,
        help="From date in YYYY-MM-DD format",
    )
    args = arg_parser.parse_args()

    asyncio.run(qdrant_ingest_flow(from_date=args.from_date))  # type: ignore[arg-type]

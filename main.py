"""Application entry point for AiraMeet AI Engine."""

import logging

from airameet_ai.core.config import get_settings
from airameet_ai.core.logger import configure_logging
from airameet_ai.data_engineering.pipeline import DataEngineeringPipeline


def main() -> int:
    """Initialize infrastructure and run the data engineering workflow."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger(__name__)
    logger.info(
        "%s starting (environment: %s)", settings.app_name, settings.environment
    )
    try:
        results = DataEngineeringPipeline(settings).run()
    except Exception:
        logger.exception("Data engineering pipeline failed")
        return 1

    logger.info(
        "Pipeline completed. Reports: %s; handoff: %s",
        results.report_paths,
        results.handoff_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

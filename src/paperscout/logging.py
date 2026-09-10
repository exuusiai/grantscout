import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure one predictable console handler for CLI and worker processes."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        stream=sys.stderr,
        force=True,
    )

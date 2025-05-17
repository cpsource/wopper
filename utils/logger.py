# logger.py
# Provides a simple file and console logging setup.

import logging
import os
from pathlib import Path

# The logger does not strictly require python-dotenv. Import it lazily so
# the module can be used even when the package is missing.
try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv  # noqa: F401
except Exception:  # pragma: no cover - missing package
    def load_dotenv(*args, **kwargs):
        return False

def get_logger(name: str, level: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    log_level = level or os.getenv("WOPPER_LOG_LEVEL", "INFO").upper()
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "wopper.log"

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger

# Local test
if __name__ == "__main__":
    log = get_logger("wopper.test")
    log.debug("This is a debug message.")
    log.info("Logger initialized successfully.")
    log.warning("This is a warning.")
    log.error("This is an error.")


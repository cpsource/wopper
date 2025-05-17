# logger.py

import logging
import os
from dotenv import load_dotenv
from pathlib import Path

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


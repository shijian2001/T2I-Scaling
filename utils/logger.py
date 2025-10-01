import logging
import sys
from typing import Optional


def setup_logger(
        log_level: int = logging.INFO,
        log_file: Optional[str] = None
) -> logging.Logger:

    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers = []

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

import os
import sys
from datetime import datetime

from loguru import logger

# Get current date
current_date = datetime.now().strftime("%Y-%m-%d")
# Log directory
LOG_DIR = f"logs/{current_date}_logs"


def setup_logger(
    project_name: str,
    log_dir=None,
    log_file=None,
    console_log_level="DEBUG",
    file_log_level="DEBUG",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
):
    log_dir = log_dir or LOG_DIR
    log_file = log_file or f"{project_name}_{current_date}.log"
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    # Remove default log handlers
    logger.remove()
    # Console log format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    # File log format
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} - {message}"
    )
    # Create project-specific log directory
    project_specific_log_dir = os.path.join(log_dir, project_name)
    os.makedirs(project_specific_log_dir, exist_ok=True)
    # Define log file path
    log_file_path = os.path.join(
        project_specific_log_dir, f"{project_name}_{{time:YYYY-MM-DD}}.log"
    )
    # Add console log handler
    logger.add(
        sys.stdout,
        format=console_format,
        level=console_log_level.upper(),
        colorize=True,
        enqueue=True,
    )
    # Add file log handler
    logger.add(
        log_file_path,
        format=file_format,
        level=file_log_level.upper(),
        rotation=rotation,
        retention=retention,
        compression=compression,
        enqueue=True,
        encoding="utf-8",
    )
    return logger

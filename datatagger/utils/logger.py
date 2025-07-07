import os
import sys
from datetime import datetime

from loguru import logger

# 获取当前日期
current_date = datetime.now().strftime("%Y-%m-%d")
# 日志目录
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
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    # 清除默认的日志处理器
    logger.remove()
    # 控制台日志格式
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    # 文件日志格式
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} - {message}"
    )
    # 创建项目特定的日志目录
    project_specific_log_dir = os.path.join(log_dir, project_name)
    os.makedirs(project_specific_log_dir, exist_ok=True)
    # 定义日志文件路径
    log_file_path = os.path.join(
        project_specific_log_dir, f"{project_name}_{{time:YYYY-MM-DD}}.log"
    )
    # 添加控制台日志处理器
    logger.add(
        sys.stdout,
        format=console_format,
        level=console_log_level.upper(),
        colorize=True,
        enqueue=True,
    )
    # 添加文件日志处理器
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

import logging
import sys
from pathlib import Path
from datetime import datetime
from . import config

def ensure_feature_dirs():
    """确保特征目录和备份目录存在"""
    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    config.FEATURE_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

def get_logger(name: str, log_dir: Path, verbose: bool = True):
    """
    获取一个配置好的 logger 实例，它会生成带时间戳的详细日志。
    """
    # 确保日志目录存在
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. 创建带时间戳的详细日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    detail_log_file = log_dir / f'{name.lower()}_{timestamp}.log'

    # 2. 为 logger 设置一个唯一的名称（基于时间戳），避免冲突
    logger = logging.getLogger(f"{name}-{timestamp}")
    logger.setLevel(logging.INFO)

    # 防止将日志消息传播到根 logger
    logger.propagate = False

    # 如果已经有处理器，则不重复添加
    if logger.hasHandlers():
        logger.handlers.clear()

    # 3. 创建详细日志的文件处理器
    detail_handler = logging.FileHandler(detail_log_file, mode='a', encoding='utf-8')
    detail_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    detail_handler.setFormatter(detail_formatter)
    logger.addHandler(detail_handler)
    
    # 4. 创建控制台处理器
    # 控制台 - INFO级别 (受verbose控制)
    if verbose:
        info_handler = logging.StreamHandler(sys.stdout)
        info_handler.setLevel(logging.INFO)
        info_handler.addFilter(lambda record: record.levelno == logging.INFO)
        info_formatter = logging.Formatter('%(message)s')
        info_handler.setFormatter(info_formatter)
        logger.addHandler(info_handler)

    # 控制台 - WARNING及以上 (始终输出)
    warn_handler = logging.StreamHandler(sys.stdout)
    warn_handler.setLevel(logging.WARNING)
    warn_formatter = logging.Formatter('%(levelname)s: %(message)s')
    warn_handler.setFormatter(warn_formatter)
    logger.addHandler(warn_handler)

    return logger, detail_log_file # 返回 logger 和日志文件路径 

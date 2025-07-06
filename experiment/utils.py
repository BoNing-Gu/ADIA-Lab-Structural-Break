import logging
import os
from datetime import datetime
from . import config

def get_timestamp():
    """获取当前时间戳字符串，格式为 YYYYMMDD_HHMMSS"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def setup_logger():
    """设置日志记录器，日志文件名将包含时间戳"""
    # 确保日志目录存在
    config.LOG_DIR.mkdir(exist_ok=True)
    
    timestamp = get_timestamp()
    log_filename = config.LOG_DIR / f'experiment_{timestamp}.log'
    
    # 获取根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 如果已经有处理器，则不重复添加
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建文件处理器
    file_handler = logging.FileHandler(filename=log_filename, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建格式化器并添加到处理器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Logger和实验目录工具

从 tool/utils_fm_duav.py 中提取的核心工具函数
"""

import os
import sys
import logging
from pathlib import Path


def get_unique_exp_dir(base_dir: str, exp_name: str) -> str:
    """
    根据基础目录和实验名称，生成一个唯一的实验文件夹路径。

    如果 `base_dir/exp_name` 已存在, 会依次尝试 `exp_name_1`,
    `exp_name_2`, ... 直到找到一个不存在的路径。

    Args:
        base_dir: 实验的根目录路径
        exp_name: 期望的实验文件夹名称

    Returns:
        一个唯一的、尚不存在的实验文件夹名称
    """
    # 将输入的字符串路径转换为Path对象，便于操作
    base_path = Path(base_dir)

    # 1. 首先检查原始名称是否存在
    original_path = base_path / exp_name
    if not original_path.exists():
        # 如果不存在，直接返回原始路径
        return exp_name

    # 2. 如果原始名称存在，开始尝试添加后缀
    counter = 1
    while True:
        # 构建新的带后缀的名称，例如 "my_experiment_1"
        suffixed_name = f"{exp_name}_{counter}"
        new_path = base_path / suffixed_name

        # 检查带后缀的路径是否存在
        if not new_path.exists():
            # 如果不存在，我们找到了唯一的路径，返回它
            return suffixed_name

        # 如果仍然存在，增加计数器，进入下一次循环
        counter += 1


def get_logger(log_file='app.log', name='my_app'):
    """
    配置一个专有的、与所有第三方库隔离的应用程序 logger。

    Args:
        log_file: 日志文件路径
        name: logger名称

    Returns:
        配置好的logger对象
    """
    # 定义日志级别和格式
    log_level = logging.INFO
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    )

    # 1. 获取我们自己的、有名字的 logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 2. 关键：设置 propagate = False
    #    这会阻止这个 logger 的消息向上传播到被 faiss 等库污染的 root logger。
    #    保证了我们日志的独立性。
    logger.propagate = False

    # 3. 检查并添加 Handlers，防止重复
    if not logger.handlers:
        # 添加文件处理器 (FileHandler)
        fh = logging.FileHandler(log_file, "a")  # 使用追加模式 'a'
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # 添加控制台处理器 (StreamHandler)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger

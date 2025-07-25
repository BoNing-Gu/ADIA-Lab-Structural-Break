#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修改后的features.py文件中的字典格式支持
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from experiment.features import (
    _save_feature_dict_file, 
    _load_feature_dict_file,
    generate_features,
    load_features,
    delete_features,
    generate_interaction_features
)
from experiment import config

def test_dict_format_functions():
    """测试字典格式的特征文件保存和加载功能"""
    print("=== 测试字典格式特征文件功能 ===")
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 100
    
    # 创建多个数据ID的测试数据
    test_feature_dict
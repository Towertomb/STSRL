#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据编码工具 - 将字典键值缩放到 0-1 范围
"""

import json
from typing import Dict, Union


def normalize_key(data: Dict[str, Union[str, int, float]], target_name: str) -> float:
    """
    识别字典中 key 的最大值，将目标 key 缩放到 0-1 之间
    
    参数:
        data: 字典，key 为数字字符串（如 "1", "2", "3"...）
        target_name: 目标值的名称（用于在字典中查找对应的 key）
    
    返回:
        缩放后的 key 值（0-1 之间的浮点数）
    
    异常:
        ValueError: 当字典为空或目标不在字典中时抛出
    """
    if not data:
        raise ValueError("字典不能为空")
    
    # 获取所有 key 并转换为数字
    numeric_keys = []
    for key in data.keys():
        try:
            numeric_keys.append(float(key))
        except (ValueError, TypeError):
            continue
    
    if not numeric_keys:
        raise ValueError("字典中没有有效的数字 key")
    
    min_key = min(numeric_keys)
    max_key = max(numeric_keys)
    
    # 查找目标值对应的 key
    target_key = None
    for key, value in data.items():
        if str(value) == str(target_name) or key == str(target_name):
            try:
                target_key = float(key)
                break
            except (ValueError, TypeError):
                continue
    
    if target_key is None:
        # 尝试直接使用 target_name 作为 key
        try:
            if str(target_name) in data:
                target_key = float(target_name)
        except (ValueError, TypeError):
            pass
    
    if target_key is None:
        raise ValueError(f"目标 '{target_name}' 不在字典中")
    
    # 缩放到 0.01-1 范围（key 不能为 0，至少为 0.01）
    if max_key == min_key:
        return 0.01
    
    normalized = 0.01 + (target_key - min_key) / (max_key - min_key) * 0.99
    return normalized


def get_key_by_name(data: Dict[str, str], target_name: str) -> str:
    """
    根据值（名字）查找对应的 key
    
    参数:
        data: 字典，value 为名字
        target_name: 目标名字
    
    返回:
        对应的 key（字符串）
    """
    for key, value in data.items():
        if value == target_name:
            return key
    raise ValueError(f"未找到名字 '{target_name}'")


def encode_data(data: Dict[str, str], target_name: str) -> dict:
    """
    主函数：给出字典和目标名字，返回缩放后的信息
    
    参数:
        data: 字典（参考 JSON 格式，key 为数字字符串，value 为名字）
        target_name: 目标名字
    
    返回:
        包含原始 key、最大值、缩放后 key 的字典
    """
    # 获取目标对应的 key
    target_key = get_key_by_name(data, target_name)
    target_key_num = float(target_key)
    
    # 获取最大 key
    max_key = max(float(k) for k in data.keys())
    min_key = min(float(k) for k in data.keys())
    
    # 计算缩放值（范围 0.01-1，key 不能为 0）
    if max_key == min_key:
        normalized = 0.01
    else:
        normalized = 0.01 + (target_key_num - min_key) / (max_key - min_key) * 0.99
    
    return {
        "target_name": target_name,
        "original_key": target_key,
        "min_key": str(min_key),
        "max_key": str(max_key),
        "normalized_key": normalized
    }


def load_json_file(file_path: str) -> Dict[str, str]:
    """
    从 JSON 文件加载数据
    
    参数:
        file_path: JSON 文件路径
    
    返回:
        字典数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    # 示例用法
    # 加载怪物数据
    monsters_file = r"G:\Disk Archive\newws\STSRL\database\monsters.json"
    
    try:
        monsters = load_json_file(monsters_file)
        print(f"加载了 {len(monsters)} 个怪物")
        
        # 测试几个怪物
        test_monsters = ["飞蝇菌子", "实验体", "女王", "机甲骑士"]
        
        for monster in test_monsters:
            result = encode_data(monsters, monster)
            print(f"\n{monster}:")
            print(f"  原始 Key: {result['original_key']}")
            print(f"  Key 范围：{result['min_key']} - {result['max_key']}")
            print(f"  缩放后：{result['normalized_key']:.4f}")
    
    except FileNotFoundError:
        print(f"文件未找到：{monsters_file}")
    except Exception as e:
        print(f"错误：{e}")

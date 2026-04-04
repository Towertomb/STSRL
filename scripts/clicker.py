"""
图像识别点击器 - 简易版

使用方法:
    from clicker import click_image
    click_image("images/start_button.png")
"""

import pyautogui
import cv2
import numpy as np
import os
from typing import Optional, Tuple

# 安全设置
pyautogui.FAILSAFE = True  # 鼠标移到屏幕角落可紧急停止
pyautogui.PAUSE = 0.1      # 操作间隔


def click_image(
    image_path: str,
    confidence: float = 0.8,
    clicks: int = 1,
    interval: float = 0.1,
    timeout: float = 5.0,
    click_offset: Tuple[int, int] = (0, 0)
) -> bool:
    """
    查找并点击指定图像
    
    Args:
        image_path: 图像文件路径
        confidence: 识别置信度 (0.5-1.0)，默认 0.8
        clicks: 点击次数，默认 1
        interval: 多次点击的间隔 (秒)，默认 0.1
        timeout: 超时时间 (秒)，默认 5.0
        click_offset: 点击偏移量 (x, y)，默认 (0, 0) 表示点击中心
    
    Returns:
        bool: 是否成功点击
    
    Raises:
        FileNotFoundError: 图像文件不存在
    
    Example:
        >>> click_image("images/start_button.png")
        True
        
        >>> click_image("images/collect.png", confidence=0.9, clicks=3)
        True
        
        >>> click_image("images/close.png", click_offset=(10, 10))
        True  # 点击中心偏移 (10, 10) 的位置
    """
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在：{image_path}")
    
    # 读取模板图像
    template = cv2.imread(image_path)
    if template is None:
        raise ValueError(f"无法读取图像：{image_path}")
    
    template_h, template_w = template.shape[:2]
    
    print(f"🔍 查找图像：{image_path} (置信度={confidence}, 超时={timeout}s)")
    
    # 在超时时间内循环查找
    import time
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        # 截取屏幕
        screenshot = pyautogui.screenshot()
        screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # 模板匹配
        result = cv2.matchTemplate(screenshot_cv, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 检查是否找到
        if max_val >= confidence:
            # 计算点击位置（默认中心）
            center_x = max_loc[0] + template_w // 2 + click_offset[0]
            center_y = max_loc[1] + template_h // 2 + click_offset[1]
            
            # 执行点击
            for i in range(clicks):
                pyautogui.click(center_x, center_y)
                if i < clicks - 1:
                    time.sleep(interval)
            
            print(f"✅ 已点击：({center_x}, {center_y})，匹配度={max_val:.2f}")
            return True
        
        # 等待下次尝试
        time.sleep(0.3)
    
    print(f"❌ 超时未找到图像：{image_path}")
    return False


def find_image(
    image_path: str,
    confidence: float = 0.8,
    timeout: float = 5.0
) -> Optional[Tuple[int, int, int, int]]:
    """
    查找图像位置（不点击）
    
    Args:
        image_path: 图像文件路径
        confidence: 识别置信度
        timeout: 超时时间
    
    Returns:
        (x, y, width, height) 或 None
    
    Example:
        >>> location = find_image("images/button.png")
        >>> if location:
        ...     x, y, w, h = location
        ...     print(f"找到于：({x}, {y}), 大小：{w}x{h}")
    """
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在：{image_path}")
    
    template = cv2.imread(image_path)
    if template is None:
        raise ValueError(f"无法读取图像：{image_path}")
    
    import time
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        screenshot = pyautogui.screenshot()
        screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        result = cv2.matchTemplate(screenshot_cv, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= confidence:
            h, w = template.shape[:2]
            print(f"✅ 找到图像：({max_loc[0]}, {max_loc[1]}), 大小：{w}x{h}, 匹配度={max_val:.2f}")
            return (max_loc[0], max_loc[1], w, h)
        
        time.sleep(0.3)
    
    print(f"❌ 未找到图像：{image_path}")
    return None


def click_until_gone(
    image_path: str,
    confidence: float = 0.8,
    max_attempts: int = 10,
    interval: float = 1.0
) -> int:
    """
    重复点击直到图像消失
    
    Args:
        image_path: 图像文件路径
        confidence: 识别置信度
        max_attempts: 最大尝试次数
        interval: 每次点击间隔
    
    Returns:
        实际点击次数
    
    Example:
        >>> # 关闭弹窗，直到关闭按钮消失
        >>> count = click_until_gone("images/close_button.png")
        >>> print(f"点击了 {count} 次")
    """
    
    count = 0
    for i in range(max_attempts):
        location = find_image(image_path, confidence, timeout=2.0)
        if location is None:
            print(f"✅ 图像已消失，共点击 {count} 次")
            return count
        
        click_image(image_path, confidence, clicks=1)
        count += 1
        time.sleep(interval)
    
    print(f"⚠️ 达到最大尝试次数 {max_attempts}")
    return count
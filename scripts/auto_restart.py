#!/usr/bin/env python3
"""
STS2 自动重开脚本

在训练过程中检测死亡状态，自动完成重开流程。

使用方法:
    from auto_restart import AutoRestarter
    
    restarter = AutoRestarter()
    restarter.start()  # 启动后台检测线程
    
    # ... 进行训练 ...
    
    restarter.stop()   # 停止检测
"""

import requests
import time
import threading
import sys
import os
from typing import Optional, Dict, Any, Callable

# 添加 scripts 目录到路径，以便导入 clicker
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clicker import click_image

# STS2 MCP API 地址
API_BASE_URL = "http://localhost:15526/api/v1/singleplayer"

# UI 图像路径（相对于 scripts 目录）
UI_IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UIimage")


class AutoRestarter:
    """
    自动重开管理器
    
    检测游戏状态，当玩家死亡时自动执行重开流程。
    """
    
    def __init__(
        self,
        api_url: str = API_BASE_URL,
        check_interval: float = 2.0,
        confidence: float = 0.9
    ):
        """
        初始化自动重开器
        
        Args:
            api_url: STS2 MCP API 地址
            check_interval: 状态检测间隔 (秒)
            confidence: 图像识别置信度
        """
        self.api_url = api_url
        self.check_interval = check_interval
        self.confidence = confidence
        
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._state_cache: Optional[Dict[str, Any]] = None
        
        # 回调函数
        self.on_death_detected: Optional[Callable] = None
        self.on_restart_complete: Optional[Callable] = None
    
    def get_game_state(self) -> Optional[Dict[str, Any]]:
        """获取游戏状态"""
        try:
            resp = requests.get(self.api_url, timeout=5)
            self._state_cache = resp.json()
            return self._state_cache
        except Exception as e:
            print(f"❌ 获取游戏状态失败：{e}")
            return self._state_cache
    
    def is_game_over(self, state: Dict[str, Any]) -> bool:
        """
        检测是否为游戏结束状态
        
        检测条件:
        - state_type == "overlay"
        - overlay.screen_type == "NGameOverScreen"
        - player.hp == 0
        """
        if state.get("state_type") != "overlay":
            return False
        
        overlay = state.get("overlay", {})
        if overlay.get("screen_type") != "NGameOverScreen":
            return False
        
        player = state.get("player", {})
        if player.get("hp", 1) > 0:
            return False
        
        return True
    
    def is_main_menu(self, state: Dict[str, Any]) -> bool:
        """检测是否在主菜单"""
        return (
            state.get("state_type") == "menu" and
            "No run in progress" in state.get("message", "")
        )
    
    def is_neow_event(self, state: Dict[str, Any]) -> bool:
        """
        检测是否为涅奥事件状态
        
        检测到后停止自动操作，交给模型处理
        """
        if state.get("state_type") != "event":
            return False
        
        event = state.get("event", {})
        return event.get("event_id") == "NEOW"
    
    def _click_ui(self, image_name: str, step_name: str) -> bool:
        """
        点击 UI 图像的辅助函数
        
        Args:
            image_name: 图像文件名（如 "ensue.png"）
            step_name: 步骤描述
        
        Returns:
            是否成功点击
        """
        image_path = os.path.join(UI_IMAGE_DIR, image_name)
        print(f"\n📌 点击'{step_name}' ({image_name})...")
        result = click_image(image_path, confidence=self.confidence, timeout=10.0)
        if result:
            print("✅ 点击成功")
        else:
            print("⚠️ 点击失败")
        time.sleep(1.0)
        return result
    
    def execute_restart_flow(self):
        """
        执行完整的重开流程
        
        流程:
        1. 点击"继续" (ensue.png)
        2. 点击"主菜单" (main_manu.png)
        3. 点击"单人模式" (single_mode.png)
        4. 点击"标准模式" (standard_mode.png)
        5. 点击"签到/开始" (check_in.png)
        6. 等待涅奥事件出现
        """
        print("\n" + "="*60)
        print("🔄 检测到死亡，开始自动重开流程")
        print("="*60)
        
        # 步骤 1: 点击"继续"
        self._click_ui("ensue.png", "继续")
        
        # 步骤 2: 点击"主菜单"
        self._click_ui("main_manu.png", "主菜单")
        
        # 步骤 3: 点击"单人模式"
        self._click_ui("single_mode.png", "单人模式")
        
        # 步骤 4: 点击"标准模式"
        self._click_ui("standard_mode.png", "标准模式")
        
        # 步骤 5: 点击"签到/开始游戏"
        self._click_ui("check_in.png", "开始游戏")
        
        # 步骤 6: 等待涅奥事件
        print("\n⏳ 等待涅奥事件出现...")
        wait_start = time.time()
        while time.time() - wait_start < 30.0:  # 最多等 30 秒
            state = self.get_game_state()
            if state and self.is_neow_event(state):
                print("✅ 涅奥事件已出现，重开流程完成！")
                print("="*60 + "\n")
                if self.on_restart_complete:
                    self.on_restart_complete()
                return True
            time.sleep(1.0)
        
        print("⚠️ 等待涅奥事件超时，但流程已结束")
        print("="*60 + "\n")
        return False
    
    def _monitor_loop(self):
        """后台监控循环"""
        print(f"\n🚀 自动重开监控已启动 (检测间隔={self.check_interval}s)")
        print("按 Ctrl+C 或调用 stop() 停止\n")
        
        last_was_death = False
        
        while self.running:
            try:
                state = self.get_game_state()
                if state is None:
                    time.sleep(self.check_interval)
                    continue
                
                # 检测死亡状态
                if self.is_game_over(state):
                    if not last_was_death:
                        print(f"\n💀 检测到死亡状态！")
                        if self.on_death_detected:
                            self.on_death_detected(state)
                        self.execute_restart_flow()
                        last_was_death = True
                else:
                    last_was_death = False
                
            except Exception as e:
                print(f"❌ 监控循环错误：{e}")
            
            time.sleep(self.check_interval)
        
        print("\n👋 自动重开监控已停止")
    
    def start(self, blocking: bool = False):
        """
        启动自动重开监控
        
        Args:
            blocking: 是否阻塞主线程
        """
        if self.running:
            print("⚠️ 自动重开已在运行中")
            return
        
        self.running = True
        
        if blocking:
            self._monitor_loop()
        else:
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
    
    def stop(self):
        """停止自动重开监控"""
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        print("\n⏹️  自动重开已停止")


# ============ 便捷函数 ============

def start_auto_restart(check_interval: float = 2.0, confidence: float = 0.9):
    """
    快速启动自动重开
    
    Example:
        from auto_restart import start_auto_restart
        restarter = start_auto_restart()
    """
    restarter = AutoRestarter(check_interval=check_interval, confidence=confidence)
    restarter.start()
    return restarter


if __name__ == "__main__":
    # 测试运行
    print("=== STS2 自动重开测试 ===\n")
    
    restarter = AutoRestarter()
    
    # 设置回调
    def on_death(state):
        print(f"📊 死亡状态：Act={state.get('run', {}).get('act')}, Floor={state.get('run', {}).get('floor')}")
    
    restarter.on_death_detected = on_death
    
    # 启动（阻塞模式，方便测试）
    try:
        restarter.start(blocking=True)
    except KeyboardInterrupt:
        restarter.stop()

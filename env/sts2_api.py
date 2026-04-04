#!/usr/bin/env python3
"""
STS2 MCP API 完整封装

包含所有可用的 API 调用示例，支持单人和多人模式。
"""

import requests
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


# ==================== 状态类型枚举 ====================

class StateType(str, Enum):
    """游戏状态类型"""
    MENU = "menu"
    MAP = "map"
    MONSTER = "monster"          # 普通战斗
    ELITE = "elite"              # 精英战斗
    BOSS = "boss"                # Boss 战斗
    COMBAT_REWARDS = "combat_rewards"  # 战斗奖励
    CARD_REWARD = "card_reward"  # 卡牌奖励选择
    REST_SITE = "rest_site"      # 休息点
    SHOP = "shop"                # 商店
    EVENT = "event"              # 事件
    CARD_SELECT = "card_select"  # 卡牌选择（升级/移除等）
    RELIC_SELECT = "relic_select"  # 遗物选择
    TREASURE = "treasure"        # 宝藏房
    HAND_SELECT = "hand_select"  # 战斗中手牌选择


# ==================== API 客户端 ====================

class STS2Client:
    """
    STS2 MCP API 完整客户端
    
    支持所有可用的 API 调用，包括单人和多人模式。
    
    使用示例:
        client = STS2Client()
        state = client.get_game_state()
        print(state['state_type'])
    """
    
    def __init__(self, host: str = "localhost", port: int = 15526, multiplayer: bool = False):
        """
        初始化客户端
        
        Args:
            host: API 主机地址
            port: API 端口（默认 15526）
            multiplayer: 是否使用多人模式端点
        """
        self.host = host
        self.port = port
        self.multiplayer = multiplayer
        self.base_url = f"http://{host}:{port}/api/v1"
        self.endpoint = "multiplayer" if multiplayer else "singleplayer"
    
    def _get(self, params: Optional[Dict] = None) -> Dict[str, Any]:
        """发送 GET 请求"""
        url = f"{self.base_url}/{self.endpoint}"
        try:
            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()
            # 检查是否是 markdown 格式
            if resp.headers.get('content-type', '').startswith('text/markdown'):
                return {"_markdown": resp.text}
            return resp.json()
        except requests.exceptions.ConnectionError:
            raise ConnectionError("无法连接到游戏，请确保 Mod 已启用且游戏正在运行")
        except Exception as e:
            raise RuntimeError(f"GET 请求失败：{e}")
    
    def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """发送 POST 请求"""
        url = f"{self.base_url}/{self.endpoint}"
        try:
            resp = requests.post(url, json=body, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            raise ConnectionError("无法连接到游戏")
        except Exception as e:
            raise RuntimeError(f"POST 请求失败：{e}")
    
    # ==================== 通用 API ====================
    
    def get_game_state(self, format: str = "json") -> Dict[str, Any]:
        """
        【通用】获取当前游戏状态
        
        返回完整的游戏状态，包括：
        - state_type: 当前界面类型
        - battle: 战斗信息（如果在战斗中）
        - player: 玩家信息
        - enemies: 敌人信息
        - map: 地图信息
        - run: 游戏进程信息
        
        Args:
            format: "json" 或 "markdown"
        
        Returns:
            游戏状态字典
        
        示例:
            state = client.get_game_state()
            print(f"当前状态：{state['state_type']}")
            print(f"楼层：{state['run']['floor']}")
        """
        return self._get({"format": format})
    
    def use_potion(self, slot: int, target: Optional[str] = None) -> Dict[str, Any]:
        """
        【通用】使用药水
        
        Args:
            slot: 药水槽位索引（从游戏状态中查看）
            target: 目标敌人 ID（如 "JAW_WORM_0"），针对敌人的药水需要
        
        Returns:
            操作结果
        
        示例:
            # 使用攻击药水
            client.use_potion(slot=0, target="JAW_WORM_0")
            
            # 使用生命药水（不需要目标）
            client.use_potion(slot=1)
        """
        body = {"action": "use_potion", "slot": slot}
        if target:
            body["target"] = target
        return self._post(body)
    
    def proceed_to_map(self) -> Dict[str, Any]:
        """
        【通用】前往地图
        
        从当前界面（奖励、休息点、商店）返回地图。
        注意：事件界面需要使用 event_choose_option 选择"继续"选项。
        
        Returns:
            操作结果
        
        示例:
            client.proceed_to_map()
        """
        return self._post({"action": "proceed"})
    
    # ==================== 战斗 API ====================
    # 适用状态：monster, elite, boss
    
    def combat_play_card(self, card_index: int, target: Optional[str] = None) -> Dict[str, Any]:
        """
        【战斗】打出手牌
        
        Args:
            card_index: 手牌索引（0-based，从游戏状态查看）
            target: 目标敌人 ID，单体卡牌需要
        
        Returns:
            操作结果
        
        示例:
            # 打出攻击牌
            client.combat_play_card(card_index=0, target="JAW_WORM_0")
            
            # 打出防御牌（不需要目标）
            client.combat_play_card(card_index=1)
        """
        body = {"action": "play_card", "card_index": card_index}
        if target:
            body["target"] = target
        return self._post(body)
    
    def combat_end_turn(self) -> Dict[str, Any]:
        """
        【战斗】结束回合
        
        Returns:
            操作结果
        
        示例:
            client.combat_end_turn()
        """
        return self._post({"action": "end_turn"})
    
    # ==================== 战斗中卡牌选择 API ====================
    # 适用状态：hand_select（卡牌效果要求选择手牌）
    
    def combat_select_card(self, card_index: int) -> Dict[str, Any]:
        """
        【战斗选择】选择手牌（用于卡牌效果）
        
        当卡牌效果要求选择手牌时使用（如"选择一张牌丢弃"）。
        
        Args:
            card_index: 可选手牌中的索引
        
        Returns:
            操作结果
        
        示例:
            client.combat_select_card(card_index=0)
        """
        return self._post({"action": "combat_select_card", "card_index": card_index})
    
    def combat_confirm_selection(self) -> Dict[str, Any]:
        """
        【战斗选择】确认选择
        
        选择完 required 数量的卡牌后确认。
        
        Returns:
            操作结果
        
        示例:
            client.combat_confirm_selection()
        """
        return self._post({"action": "combat_confirm_selection"})
    
    # ==================== 奖励 API ====================
    # 适用状态：combat_rewards, card_reward
    
    def rewards_claim(self, reward_index: int) -> Dict[str, Any]:
        """
        【奖励】领取战斗奖励
        
        Args:
            reward_index: 奖励索引（从右到左领取可保持索引稳定）
        
        Returns:
            操作结果
        
        示例:
            # 领取卡牌奖励
            client.rewards_claim(reward_index=2)
        """
        return self._post({"action": "claim_reward", "index": reward_index})
    
    def rewards_pick_card(self, card_index: int) -> Dict[str, Any]:
        """
        【奖励】选择卡牌奖励
        
        Args:
            card_index: 可选卡牌中的索引
        
        Returns:
            操作结果
        
        示例:
            client.rewards_pick_card(card_index=0)
        """
        return self._post({"action": "select_card_reward", "card_index": card_index})
    
    def rewards_skip_card(self) -> Dict[str, Any]:
        """
        【奖励】跳过卡牌奖励
        
        Returns:
            操作结果
        
        示例:
            client.rewards_skip_card()
        """
        return self._post({"action": "skip_card_reward"})
    
    # ==================== 地图 API ====================
    # 适用状态：map
    
    def map_choose_node(self, node_index: int) -> Dict[str, Any]:
        """
        【地图】选择地图节点
        
        Args:
            node_index: 可前往节点的索引
        
        Returns:
            操作结果
        
        示例:
            # 选择第一个可前往的节点
            client.map_choose_node(node_index=0)
        """
        return self._post({"action": "choose_map_node", "index": node_index})
    
    # ==================== 休息点 API ====================
    # 适用状态：rest_site
    
    def rest_choose_option(self, option_index: int) -> Dict[str, Any]:
        """
        【休息点】选择休息选项
        
        Args:
            option_index: 选项索引（休息/锻造等）
        
        Returns:
            操作结果
        
        示例:
            # 选择休息
            client.rest_choose_option(option_index=0)
        """
        return self._post({"action": "choose_rest_option", "index": option_index})
    
    # ==================== 商店 API ====================
    # 适用状态：shop
    
    def shop_purchase(self, item_index: int) -> Dict[str, Any]:
        """
        【商店】购买商品
        
        Args:
            item_index: 商品索引
        
        Returns:
            操作结果
        
        示例:
            client.shop_purchase(item_index=0)
        """
        return self._post({"action": "shop_purchase", "index": item_index})
    
    # ==================== 事件 API ====================
    # 适用状态：event
    
    def event_choose_option(self, option_index: int) -> Dict[str, Any]:
        """
        【事件】选择事件选项
        
        Args:
            option_index: 选项索引
        
        Returns:
            操作结果
        
        示例:
            client.event_choose_option(option_index=0)
        """
        return self._post({"action": "choose_event_option", "index": option_index})
    
    def event_advance_dialogue(self) -> Dict[str, Any]:
        """
        【事件】推进对话
        
        用于古代事件等需要点击继续的对话。
        
        Returns:
            操作结果
        
        示例:
            client.event_advance_dialogue()
        """
        return self._post({"action": "advance_dialogue"})
    
    # ==================== 卡牌选择 API ====================
    # 适用状态：card_select（升级/移除/转化卡牌）
    
    def deck_select_card(self, card_index: int) -> Dict[str, Any]:
        """
        【卡牌选择】选择/取消选择卡牌
        
        用于卡组中选择卡牌（升级、移除、转化等）。
        
        Args:
            card_index: 卡牌索引
        
        Returns:
            操作结果
        
        示例:
            # 选择一张牌升级
            client.deck_select_card(card_index=0)
        """
        return self._post({"action": "select_card", "index": card_index})
    
    def deck_confirm_selection(self) -> Dict[str, Any]:
        """
        【卡牌选择】确认选择
        
        Returns:
            操作结果
        
        示例:
            client.deck_confirm_selection()
        """
        return self._post({"action": "confirm_selection"})
    
    def deck_cancel_selection(self) -> Dict[str, Any]:
        """
        【卡牌选择】取消选择
        
        Returns:
            操作结果
        
        示例:
            client.deck_cancel_selection()
        """
        return self._post({"action": "cancel_selection"})
    
    # ==================== 遗物选择 API ====================
    # 适用状态：relic_select
    
    def relic_select(self, relic_index: int) -> Dict[str, Any]:
        """
        【遗物选择】选择遗物
        
        Args:
            relic_index: 遗物索引
        
        Returns:
            操作结果
        
        示例:
            client.relic_select(relic_index=0)
        """
        return self._post({"action": "select_relic", "index": relic_index})
    
    def relic_skip(self) -> Dict[str, Any]:
        """
        【遗物选择】跳过遗物选择
        
        Returns:
            操作结果
        
        示例:
            client.relic_skip()
        """
        return self._post({"action": "skip_relic_selection"})
    
    # ==================== 宝藏房 API ====================
    # 适用状态：treasure
    
    def treasure_claim_relic(self, relic_index: int) -> Dict[str, Any]:
        """
        【宝藏】领取遗物
        
        Args:
            relic_index: 遗物索引
        
        Returns:
            操作结果
        
        示例:
            client.treasure_claim_relic(relic_index=0)
        """
        return self._post({"action": "claim_treasure_relic", "index": relic_index})
    
    # ==================== 卡牌包选择 API ====================
    # 适用状态：bundle_select
    
    def bundle_select(self, bundle_index: int) -> Dict[str, Any]:
        """
        【卡牌包选择】选择卡牌包
        
        用于开局选择初始卡牌包，或其他卡牌包选择场景。
        
        Args:
            bundle_index: 卡牌包索引（0-based）
        
        Returns:
            操作结果
        
        示例:
            # 选择第一个卡包
            client.bundle_select(bundle_index=0)
        """
        return self._post({"action": "bundle_select", "index": bundle_index})


# ==================== 多人模式 API ====================

class STS2MultiplayerClient(STS2Client):
    """
    STS2 多人模式 API 客户端
    
    所有操作都会通过多人端点，支持投票机制。
    
    使用示例:
        client = STS2MultiplayerClient()
        state = client.get_game_state()
    """
    
    def __init__(self, host: str = "localhost", port: int = 15526):
        super().__init__(host, port, multiplayer=True)
    
    def mp_combat_end_turn(self) -> Dict[str, Any]:
        """
        【多人战斗】提交结束回合投票
        
        多人模式下，结束回合需要所有玩家投票。
        """
        return self._post({"action": "end_turn"})
    
    def mp_combat_undo_end_turn(self) -> Dict[str, Any]:
        """
        【多人战斗】撤销结束回合投票
        
        Returns:
            操作结果
        """
        return self._post({"action": "undo_end_turn"})
    
    def mp_map_vote(self, node_index: int) -> Dict[str, Any]:
        """
        【多人地图】投票选择地图节点
        
        Returns:
            操作结果
        """
        return self._post({"action": "choose_map_node", "index": node_index})
    
    def mp_treasure_claim_relic(self, relic_index: int) -> Dict[str, Any]:
        """
        【多人宝藏】竞标遗物
        
        多人模式下，这是竞标而非直接领取。
        
        Returns:
            操作结果
        """
        return self._post({"action": "claim_treasure_relic", "index": relic_index})


# ==================== 工具函数 ====================

def print_state_summary(state: Dict[str, Any]) -> None:
    """
    打印游戏状态摘要
    
    Args:
        state: get_game_state() 返回的状态字典
    """
    print("=" * 50)
    print(f"状态类型：{state.get('state_type', 'unknown')}")
    
    run = state.get('run', {})
    print(f"楼层：{run.get('floor', '?')}")
    print(f"层数：{run.get('level', '?')}")
    
    battle = state.get('battle', {})
    if battle:
        player = battle.get('player', {})
        print(f"HP: {player.get('hp', '?')}/{player.get('max_hp', '?')}")
        print(f"格挡：{player.get('block', 0)}")
        print(f"能量：{player.get('energy', 0)}")
        
        enemies = battle.get('enemies', [])
        print(f"敌人数量：{len(enemies)}")
        for i, enemy in enumerate(enemies):
            print(f"  - {enemy.get('name', '?')}: {enemy.get('hp', '?')} HP")
    
    print("=" * 50)


# ==================== 测试入口 ====================

if __name__ == "__main__":
    print("🎮 STS2 API 测试")
    print("=" * 50)
    
    client = STS2Client()
    
    try:
        state = client.get_game_state()
        print_state_summary(state)
        print("\n✅ 连接成功！")
    except Exception as e:
        print(f"\n❌ 连接失败：{e}")
        print("\n请确保:")
        print("1. 游戏正在运行")
        print("2. STS2_MCP mod 已启用")
        print("3. 已开始一局游戏")

#!/usr/bin/env python3
"""
STS2 Gymnasium 环境包装器 - 修复版

修复了动作空间问题，使用 Discrete 而非 MultiDiscrete。
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import time
from typing import Dict, Any, Optional, Tuple, List
from env.sts2_api import STS2Client


class STS2Env(gym.Env):
    """
    STS2 强化学习环境 - 修复版
    
    使用 Discrete 动作空间，将动作编码为单一整数：
    - 0-9: 出牌 0-9
    - 10-19: 用药 0-9
    - 20: 结束回合
    - 21: 跳过
    
    这样 PPO 可以正确处理动作概率分布。
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 15526,
        max_hand_size: int = 10,
        max_enemies: int = 10,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.client = STS2Client(host, port)
        self.max_hand_size = max_hand_size
        self.max_enemies = max_enemies
        self.render_mode = render_mode
        
        # 状态跟踪
        self.last_hp = None
        self.last_floor = None
        self.last_hand_size = None
        
        # ==================== 观测空间 ====================
        player_dim = 5
        hand_dim = max_hand_size * 4
        enemy_dim = max_enemies * 3
        obs_dim = player_dim + hand_dim + enemy_dim + 10
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # ==================== 动作空间 (修复) ====================
        # 使用 Discrete 而非 MultiDiscrete
        # 0-9: 出牌，10-19: 用药，20: 结束回合，21: 跳过
        self.action_space = spaces.Discrete(22)
        
        self.card_values = {
            "Strike": 1.5,
            "Defend": 1.2,
            "Bash": 3.0,
        }
        
        # 日志列表
        self.event_log = []
    
    def log_event(self, msg: str):
        """记录事件日志"""
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.event_log.append(entry)
        try:
            print(entry)
        except:
            print(entry.encode('gbk', errors='replace').decode('gbk'))
    
    def _decode_action(self, action: int, state: Dict) -> Tuple[str, int, Optional[str]]:
        """
        将离散动作解码为具体操作
        
        Returns:
            (action_type, param, target)
        """
        state_type = state.get('state_type', '')
        
        # 战斗状态
        if state_type in ['monster', 'elite', 'boss']:
            if action < 10:  # 0-9: 出牌
                return ("play_card", action, None)
            elif action < 20:  # 10-19: 用药
                return ("use_potion", action - 10, None)
            elif action == 20:  # 20: 结束回合
                return ("end_turn", 0, None)
            else:  # 21: 跳过
                return ("wait", 0, None)
        
        # 奖励状态
        elif state_type in ['combat_rewards', 'rewards']:
            # 检查是否可以前进
            rewards_data = state.get('rewards', {})
            can_proceed = rewards_data.get('can_proceed', False)
            items = rewards_data.get('items', [])
            
            if can_proceed and action >= 20:
                # 动作 20+：跳过奖励直接前进
                return ("proceed", 0, None)
            elif items and len(items) > 0:
                # 动作 0-19：选择奖励
                return ("claim_reward", action % len(items), None)
            else:
                # 没有奖励，前进
                return ("proceed", 0, None)
        
        # 卡牌奖励
        elif state_type == 'card_reward':
            if action < 10:
                return ("pick_card", action % 3, None)
            else:
                return ("skip_card", 0, None)
        
        # 地图
        elif state_type == 'map':
            return ("choose_node", action % 3, None)
        
        # 休息点
        elif state_type == 'rest_site':
            if action < 10:
                return ("rest_option", action % 2, None)
            else:
                return ("proceed", 0, None)
        
        # 商店
        elif state_type == 'shop':
            if action < 10:
                return ("purchase", action % 5, None)
            else:
                return ("proceed", 0, None)
        
        # 事件
        elif state_type == 'event':
            if action < 20:
                return ("event_option", action % 3, None)
            else:
                return ("advance_dialogue", 0, None)
        
        # 卡牌选择（升级/移除/转化）
        elif state_type == 'card_select':
            # 在 card_select 状态下，0-9 选择卡牌，10+ 确认
            if action < 10:
                return ("select_card", action, None)
            else:
                return ("confirm", 0, None)
        
        # 宝藏房
        elif state_type == 'treasure':
            # 0-9: 选择遗物，10+: 前进
            if action < 10:
                return ("claim_relic", action, None)
            else:
                return ("proceed", 0, None)
        
        # 覆盖层/菜单状态
        elif state_type in ['overlay', 'menu']:
            return ("wait", 0, None)
        
        # 默认
        return ("wait", 0, None)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        print("\n等待游戏状态...")
        try:
            while True:
                state = self.client.get_game_state()
                if state.get('state_type') not in ['menu', 'unknown']:
                    print(f"检测到状态：{state['state_type']}")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
        state = self.client.get_game_state()
        
        self.last_hp = self._get_player_hp(state)
        self.last_floor = state.get('run', {}).get('floor', 0)
        self.last_hand_size = len(state.get('player', {}).get('hand', []))
        
        obs = self._get_observation(state)
        info = {"state": state}
        
        return obs, info
    
    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行动作"""
        reward = 0.0
        terminated = False
        info = {}
        
        try:
            state = self.client.get_game_state()
            state_type = state.get('state_type', '')
            
            # 解码动作
            action_type, param, target = self._decode_action(int(action), state)
            
            # 打印动作信息
            action_desc = f"动作：{action} → {action_type}"
            if param != 0:
                action_desc += f" (参数={param})"
            self.log_event(action_desc)
            
            # 记录出牌前的手牌数
            old_hand_size = len(state.get('player', {}).get('hand', []))
            old_energy = state.get('player', {}).get('energy', 0)
            old_battle_turn = state.get('battle', {}).get('turn', '')
            
            # ==================== 执行动作 ====================
            if state_type in ['monster', 'elite', 'boss']:
                # 检查是否在玩家回合
                battle = state.get('battle', {})
                turn = battle.get('turn', '')
                
                if turn != 'player':
                    self.log_event(f"  等待：敌方回合")
                    time.sleep(1)
                    # 重新获取状态
                    state = self.client.get_game_state()
                    battle = state.get('battle', {})
                    turn = battle.get('turn', '')
                
                if action_type == "play_card":
                    reward = self._play_card(param, state)
                elif action_type == "use_potion":
                    reward = self._use_potion(param, state)
                elif action_type == "end_turn":
                    result = self.client.combat_end_turn()
                    info['action_result'] = result
                    time.sleep(0.3)
                # wait: 什么也不做
            
            elif state_type in ['combat_rewards', 'rewards']:
                # 检查是否有奖励可领取
                rewards_data = state.get('rewards', {})
                items = rewards_data.get('items', [])
                can_proceed = rewards_data.get('can_proceed', False)
                
                if action_type == "proceed" and can_proceed:
                    # 模型选择跳过奖励直接前进
                    result = self.client.proceed_to_map()
                    info['action_result'] = result
                    info['skipped_rewards'] = True
                    self.log_event("跳过奖励，直接前进")
                    time.sleep(0.3)
                elif items and len(items) > 0:
                    # 有奖励，检查奖励类型
                    reward_item = items[param % len(items)]
                    reward_type = reward_item.get('type', '')
                    
                    # 如果是药水奖励，检查药水栏是否已满
                    if reward_type == 'potion':
                        player = state.get('player', {})
                        potions = player.get('potions', [])
                        max_potions = 3  # STS2 最大药水栏位
                        
                        if len(potions) >= max_potions:
                            # 药水栏已满，自动前进（不领取药水）
                            result = self.client.proceed_to_map()
                            info['action_result'] = result
                            info['auto_proceed'] = True
                            self.log_event(f"药水栏已满 ({len(potions)}/{max_potions})，跳过药水奖励")
                        else:
                            # 药水栏有空位，领取药水
                            result = self.client.rewards_claim(param % len(items))
                            info['action_result'] = result
                    else:
                        # 非药水奖励（金币、卡牌等），正常领取
                        result = self.client.rewards_claim(param % len(items))
                        info['action_result'] = result
                    
                    time.sleep(0.3)
                    
                    # 检查是否进入卡牌选择状态（卡牌奖励）
                    if reward_type == 'card':
                        time.sleep(0.5)
                        new_state = self.client.get_game_state()
                        if new_state.get('state_type') == 'card_reward':
                            # 自动选择第一张牌（简化策略）
                            cards = new_state.get('cards', [])
                            if cards:
                                self.client.rewards_pick_card(0)
                                self.log_event(f"选择卡牌：{cards[0].get('name', '?')}")
                                time.sleep(0.5)
                else:
                    # 没有奖励（如白银熔炉的空宝箱），自动前进
                    result = self.client.proceed_to_map()
                    info['action_result'] = result
                    info['auto_proceed'] = True
                time.sleep(0.3)
            
            elif state_type == 'card_reward':
                if action_type == "pick_card":
                    result = self.client.rewards_pick_card(param)
                else:
                    result = self.client.rewards_skip_card()
                info['action_result'] = result
                time.sleep(0.3)
            
            # ==================== 宝藏房 ====================
            elif state_type == 'treasure':
                # 宝藏房状态，检查是否有遗物可选
                treasure_data = state.get('treasure', {})
                relics = treasure_data.get('relics', [])
                can_proceed = treasure_data.get('can_proceed', False)
                
                if action_type == "claim_relic":
                    if relics and len(relics) > 0:
                        # 有遗物可选，选择指定遗物
                        result = self.client.treasure_claim_relic(param % len(relics))
                        info['action_result'] = result
                        self.log_event(f"选择宝藏遗物：{relics[param % len(relics)].get('name', '?')}")
                        time.sleep(0.5)
                        
                        # 领取后自动前进
                        result = self.client.proceed_to_map()
                        info['action_result'] = result
                        self.log_event("领取遗物后自动前进")
                    else:
                        # 没有遗物，前进
                        result = self.client.proceed_to_map()
                        info['action_result'] = result
                        info['auto_proceed'] = True
                        self.log_event("宝藏房为空，自动前进")
                elif action_type == "proceed" or can_proceed:
                    # 直接前进
                    result = self.client.proceed_to_map()
                    info['action_result'] = result
                    info['auto_proceed'] = True
                    self.log_event("宝藏房自动前进")
                
                time.sleep(0.3)
            
            elif state_type == 'map':
                result = self.client.map_choose_node(param)
                info['action_result'] = result
                time.sleep(0.5)
            
            elif state_type == 'rest_site':
                if action_type == "rest_option":
                    result = self.client.rest_choose_option(param)
                else:
                    result = self.client.proceed_to_map()
                info['action_result'] = result
                time.sleep(0.3)
            
            elif state_type == 'shop':
                if action_type == "purchase":
                    result = self.client.shop_purchase(param)
                else:
                    result = self.client.proceed_to_map()
                info['action_result'] = result
                time.sleep(0.3)
            
            elif state_type == 'event':
                if action_type == "event_option":
                    result = self.client.event_choose_option(param)
                else:
                    result = self.client.event_advance_dialogue()
                info['action_result'] = result
                time.sleep(0.3)
            
            # ==================== 卡牌选择（升级/移除/转化） ====================
            elif state_type == 'card_select':
                if action_type == "select_card":
                    # 选择卡牌
                    result = self.client.deck_select_card(param % 10)
                    info['action_result'] = result
                elif action_type == "confirm":
                    # 确认选择
                    result = self.client.deck_confirm_selection()
                    info['action_result'] = result
                elif action_type == "cancel":
                    # 取消选择
                    result = self.client.deck_cancel_selection()
                    info['action_result'] = result
                else:
                    # 默认：选择第一张牌（升级场景通常不需要确认）
                    result = self.client.deck_select_card(param % 10)
                    info['action_result'] = result
                time.sleep(0.5)
            
            # ==================== 其他状态（overlay、menu 等） ====================
            elif state_type == 'overlay':
                # 覆盖层状态（如死亡界面、事件覆盖等），等待状态变化
                self.log_event("  等待：覆盖层状态")
                time.sleep(1)
            elif state_type == 'menu':
                # 菜单状态，等待新游戏开始
                self.log_event("  等待：菜单状态，请开始新游戏")
                time.sleep(2)
            else:
                # 未知状态
                time.sleep(0.5)
            
            # 获取新状态
            new_state = self.client.get_game_state()
            obs = self._get_observation(new_state)
            
            # 计算奖励
            reward += self._calculate_reward(state, new_state, old_hand_size, old_energy, old_battle_turn)
            
            # 检查死亡（只在战斗状态检查，避免游戏结束间隙误判）
            new_state_type = new_state.get('state_type', '')
            if new_state_type in ['monster', 'elite', 'boss']:
                current_hp = self._get_player_hp(new_state)
                if current_hp is not None and current_hp <= 0:
                    reward -= 100
                    terminated = True
                    info['reason'] = 'death'
                    self.log_event("角色死亡！")
            
            info['state'] = new_state
            
            # 打印奖励摘要
            if reward != 0:
                breakdown = self._last_reward_breakdown if hasattr(self, '_last_reward_breakdown') else {}
                if breakdown:
                    self.log_event(f"奖励：{reward:.2f} = {breakdown}")
                else:
                    self.log_event(f"奖励：{reward:.2f}")
            
        except Exception as e:
            print(f"环境 step 错误：{e}")
            obs = self.observation_space.sample()
            reward = -1.0
            info['error'] = str(e)
        
        return obs, reward, terminated, False, info
    
    def _calculate_reward(self, old_state: Dict, new_state: Dict, 
                          old_hand_size: int, old_energy: int, old_turn: str) -> float:
        """计算奖励"""
        reward = 0.0
        breakdown = {}
        
        # HP 变化
        old_hp = self.last_hp or 0
        new_hp = self._get_player_hp(new_state) or 0
        hp_diff = new_hp - old_hp
        if hp_diff < 0:
            reward += hp_diff * 0.2
            breakdown['hp_loss'] = hp_diff * 0.2
        self.last_hp = new_hp
        
        # 楼层变化
        old_floor = self.last_floor or 0
        new_floor = new_state.get('run', {}).get('floor', 0)
        if new_floor > old_floor:
            reward += 5.0
            breakdown['floor_clear'] = 5.0
            self.last_floor = new_floor
        
        # 战斗胜利
        old_type = old_state.get('state_type', '')
        new_type = new_state.get('state_type', '')
        if old_type in ['monster', 'elite', 'boss'] and new_type == 'combat_rewards':
            reward += 50.0
            breakdown['battle_win'] = 50.0
        
        # 造成伤害
        old_enemies = old_state.get('battle', {}).get('enemies', [])
        new_enemies = new_state.get('battle', {}).get('enemies', [])
        old_hp_sum = sum(e.get('hp', 0) for e in old_enemies)
        new_hp_sum = sum(e.get('hp', 0) for e in new_enemies)
        damage = old_hp_sum - new_hp_sum
        if damage > 0:
            reward += damage * 0.2
            breakdown['damage'] = damage * 0.2
        
        # 击杀敌人
        if len(new_enemies) < len(old_enemies):
            killed = len(old_enemies) - len(new_enemies)
            reward += killed * 15.0
            breakdown['kill'] = killed * 15.0
        
        # 出牌奖励
        new_hand_size = len(new_state.get('player', {}).get('hand', []))
        if new_hand_size < old_hand_size:
            cards = old_hand_size - new_hand_size
            reward += cards * 1.0
            breakdown['play_card'] = cards * 1.0
        
        # 有能量却结束回合的惩罚
        if old_energy > 0 and old_type in ['monster', 'elite', 'boss']:
            new_turn = new_state.get('battle', {}).get('turn', '')
            if old_turn == 'player' and new_turn != 'player':
                reward -= 0.5
                breakdown['wasted_energy'] = -0.5
        
        self._last_reward_breakdown = breakdown
        return reward
    
    def _get_observation(self, state: Dict) -> np.ndarray:
        """将游戏状态转换为观测向量"""
        obs = []
        
        # 玩家状态 (player 在顶层)
        player = state.get('player', {})
        
        hp = player.get('hp', 0)
        max_hp = player.get('max_hp', 1)
        obs.append(hp / max(max_hp, 1))
        obs.append(player.get('energy', 0))
        obs.append(player.get('block', 0))
        obs.append(player.get('gold', 0))
        obs.append(state.get('run', {}).get('floor', 0))
        
        # 手牌
        hand = player.get('hand', [])
        for i in range(self.max_hand_size):
            if i < len(hand):
                card = hand[i]
                type_enc = {'Attack': 1, 'Skill': 2, 'Power': 3}.get(card.get('type', ''), 0)
                cost = card.get('cost', 0)
                if cost == 'X': cost = 0
                try:
                    cost = int(cost)
                except:
                    cost = 0
                value = self._estimate_card_value(card)
                can_play = 1 if card.get('can_play', False) else 0
                obs.extend([type_enc, cost, value, can_play])
            else:
                obs.extend([0, 0, 0, 0])
        
        # 敌人状态
        enemies = state.get('battle', {}).get('enemies', [])
        for i in range(self.max_enemies):
            if i < len(enemies):
                enemy = enemies[i]
                obs.append(enemy.get('hp', 0) / max(enemy.get('max_hp', 1), 1))
                obs.append(self._get_enemy_intent(enemy))
                obs.append(len(enemy.get('debuffs', [])))
            else:
                obs.extend([0, 0, 0])
        
        # 预留
        obs.extend([0] * 10)
        
        return np.array(obs, dtype=np.float32)
    
    def _estimate_card_value(self, card: Dict) -> float:
        name = card.get('name', '')
        base = self.card_values.get(name, 1.0)
        desc = card.get('description', '')
        if '造成' in desc: base += 1.0
        if '格挡' in desc: base += 0.8
        if '抽' in desc: base += 1.5
        return base
    
    def _get_enemy_intent(self, enemy: Dict) -> float:
        total = 0
        for intent in enemy.get('intents', []):
            if intent.get('type') == 'Attack':
                try:
                    total += int(intent.get('label', 0))
                except:
                    pass
        return total
    
    def _get_player_hp(self, state: Dict) -> Optional[int]:
        player = state.get('player', {})
        return player.get('hp')
    
    def _play_card(self, card_index: int, state: Dict) -> float:
        # player 在顶层，不在 battle 里面！
        player = state.get('player', {})
        hand = player.get('hand', []) if player else []
        
        # 检查索引是否越界
        if card_index >= len(hand):
            self.log_event(f"  出牌失败：索引{card_index}越界（手牌只有{len(hand)}张）")
            return -0.5
        
        card = hand[card_index]
        
        # 检查卡牌是否可出
        if not card.get('can_play', False):
            cost = card.get('cost', '?')
            self.log_event(f"  出牌失败：{card.get('name')} 不可出 (费用={cost}, can_play=False)")
            return -0.5
        
        target = None
        if card.get('target_type') == 'AnyEnemy':
            enemies = state.get('battle', {}).get('enemies', [])
            if enemies:
                target = min(enemies, key=lambda e: e.get('hp', 999)).get('entity_id')
        
        try:
            result = self.client.combat_play_card(int(card_index), target)
            self.log_event(f"  出牌成功：{card.get('name')}")
            return 0.1
        except Exception as e:
            self.log_event(f"  出牌失败：{e}")
            return -0.5
    
    def _use_potion(self, slot: int, state: Dict) -> float:
        try:
            player = state.get('player', {})
            potions = player.get('potions', [])
            if slot >= len(potions):
                return -0.5
            
            potion = potions[slot]
            target = None
            if '敌人' in potion.get('description', ''):
                enemies = state.get('battle', {}).get('enemies', [])
                if enemies:
                    target = enemies[0].get('entity_id')
            
            # 记录用药前 HP
            hp_before = player.get('hp', 0)
            
            self.client.use_potion(int(slot), target)
            
            # 用药基础奖励
            reward = 1.0
            
            
            return reward
        except Exception as e:
            print(f"用药失败：{e}")
            return -0.5
    
    def render(self):
        if self.render_mode == "human":
            state = self.client.get_game_state()
            from env.sts2_api import print_state_summary
            print_state_summary(state)


# 注册环境
gym.register(
    id="STS2-v1",  # 新版本号
    entry_point="sts2_env_fixed:STS2EnvFixed",
    max_episode_steps=10000,
)


if __name__ == "__main__":
    print("测试修复版环境...")
    env = STS2EnvFixed()
    
    obs, info = env.reset()
    print(f"观测形状：{obs.shape}")
    print(f"动作空间：{env.action_space}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.2f}")
        if terminated:
            break
    
    env.close()
    print("测试完成")

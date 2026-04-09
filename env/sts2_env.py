import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import time
import json
from typing import Dict, Any, Optional, Tuple, List
from env.sts2_api import STS2Client


class STS2Env(gym.Env):
    """
    
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
        # 状态类型 (1 维) + 玩家 (15 维) + 手牌/卡牌选择/商店 (211 维) + 敌人/事件 (170 维) + 药水 (6 维) = 403 维
        # 敌人维度：10 个 × 17 维 (基础 7 维：ID+HP+2 个意图×2 维+status 数 + 5 个 status×2 维)
        state_type_dim = 1            # 状态类型编码
        player_base_dim = 5           # HP 比例 + 能量 + 格挡 + 金币 + 楼层
        player_status_dim = 5 * 2     # 最多 5 个 status，每个 2 维 (status ID + 类型)
        player_dim = player_base_dim + player_status_dim
        self.max_cards = 35           # 最大卡牌数量（card_select/shop 状态）
        card_select_dim = 1 + self.max_cards * 6  # card_select: screen_type(1 维) + 35 张牌×6 维=211 维
        shop_dim = 1 + self.max_cards * 6  # shop: screen_type(1 维) + 35 个商品×6 维=211 维
        enemy_base_dim = 7            # 敌人 ID + HP + 意图 1(类型 + 数值) + 意图 2(类型 + 数值) + status 数
        enemy_status_dim = 5 * 2      # 最多 5 个 status，每个 2 维
        enemy_dim = max_enemies * (enemy_base_dim + enemy_status_dim)  # 10 × 17 = 170 维
        event_base_dim = 3            # 事件 ID + is_ancient + in_dialogue
        event_option_dim = 5 * 5      # 最多 5 个选项，每个 5 维 (index+desc 数 +is_locked+is_proceed+was_chosen)
        event_dim = event_base_dim + event_option_dim  # 28 维
        potion_dim = 6                # 最多 6 瓶药水，每瓶 1 维（药水 ID）
        obs_dim = state_type_dim + player_dim + max(card_select_dim, shop_dim) + max(enemy_dim, event_dim) + potion_dim  # 403 维
        
        self.max_desc_number = 50     # 描述数字最大归一化值
        self.max_event_id = 56        # events.json 最大 ID
        self.max_relic_id = 200       # relics.json 最大 ID（预估）
        
        # card_select screen_type 编码
        self.card_select_type_map = {
            'upgrade': 1,
            'select': 2,
            'transform': 3,
            'unknown': 0
        }
        
        # shop 商品类型编码
        self.shop_item_type_map = {
            'card': 1,
            'relic': 2,
            'potion': 3,
            'card_removal': 4,
            'unknown': 0
        }
        
        # 状态类型编码
        self.state_type_map = {
            'menu': 0,
            'unknown': 1,
            'monster': 2,
            'elite': 3,
            'boss': 4,
            'hand_select': 5,
            'rewards': 6,
            'card_reward': 7,
            'map': 8,
            'event': 9,
            'rest_site': 10,
            'shop': 11,
            'treasure': 12,
            'card_select': 13,
            'bundle_select': 14,
            'relic_select': 15,
            'crystal_sphere': 16,
            'overlay': 17
        }
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # 加载数据库编码
        self.card_db = self._load_card_db()
        self.monster_db = self._load_monster_db()
        self.potion_db = self._load_potion_db()
        self.status_db = self._load_status_db()
        self.event_db = self._load_event_db()
        self.relic_db = self._load_relic_db()
        
        # Status 类型编码
        self.status_type_map = {
            'Buff': 1,
            'Debuff': 2,
            'Unknown': 0
        }
        
        # ==================== 动作空间 (修复) ====================
        # 使用 Discrete 而非 MultiDiscrete
        # 0-9: 出牌，10-19: 用药，20: 结束回合，21: 跳过
        self.action_space = spaces.Discrete(22)
        
        # 归一化最大值
        self.max_energy = 5
        self.max_block = 50
        self.max_gold = 100
        self.max_floor = 60
        self.max_intent_damage = 50  # 攻击意图最大伤害
        self.max_card_id = 577  # card.json 最大 ID
        self.max_monster_id = 107  # monsters.json 最大 ID
        self.max_potion_id = 63  # potions.json 最大 ID
        self.max_status_id = 259  # status.json 最大 ID
        
        # 意图类型编码
        self.intent_type_map = {
            'Attack': 1,
            'Defend': 2,
            'Buff': 3,
            'Debuff': 4,
            'Sleep': 5,
            'Unknown': 0
        }
        
        # 日志列表
        self.event_log = []
    
    def _load_card_db(self) -> Dict[str, int]:
        """加载卡牌数据库，建立名称到 ID 的映射"""
        import json
        import os
        db_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'card.json')
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 反转映射：名称 -> ID
            return {name: int(id) for id, name in data.items()}
        except:
            return {}
    
    def _load_monster_db(self) -> Dict[str, int]:
        """加载怪物数据库，建立名称到 ID 的映射"""
        import json
        import os
        db_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'monsters.json')
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {name: int(id) for id, name in data.items()}
        except:
            return {}
    
    def _load_potion_db(self) -> Dict[str, int]:
        """加载药水数据库，建立名称到 ID 的映射"""
        import json
        import os
        db_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'potions.json')
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {name: int(id) for id, name in data.items()}
        except:
            return {}
    
    def _load_status_db(self) -> Dict[str, int]:
        """加载 status 数据库，建立名称到 ID 的映射"""
        import json
        import os
        db_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'status.json')
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # status.json 格式：{"0": "名称 1", "1": "名称 2", ...}
            # 反转映射：名称 -> ID
            return {name: int(id) for id, name in data.items()}
        except:
            return {}
    
    def _load_event_db(self) -> Dict[str, int]:
        """加载 event 数据库，建立名称到 ID 的映射"""
        import json
        import os
        db_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'events.json')
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # events.json 格式：{"0": "名称 1", "1": "名称 2", ...}
            # 反转映射：名称 -> ID
            return {name: int(id) for id, name in data.items()}
        except:
            return {}
    
    def _load_relic_db(self) -> Dict[str, int]:
        """加载 relic 数据库，建立名称到 ID 的映射"""
        import json
        import os
        db_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'relics.json')
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # relics.json 格式：{"0": "名称 1", "1": "名称 2", ...}
            # 反转映射：名称 -> ID
            return {name: int(id) for id, name in data.items()}
        except:
            return {}
    
    def _get_event_id(self, event_name: str) -> float:
        """获取事件 ID 并归一化"""
        event_id = self.event_db.get(event_name, 0)
        return event_id / self.max_event_id
    
    def _get_relic_id(self, relic_name: str) -> float:
        """获取遗物 ID 并归一化"""
        relic_id = self.relic_db.get(relic_name, 0)
        return relic_id / self.max_relic_id
    
    def _get_card_id(self, card_name: str) -> float:
        """获取卡牌 ID 并归一化"""
        card_id = self.card_db.get(card_name, 0)
        return card_id / self.max_card_id
    
    def _get_monster_id(self, monster_name: str) -> float:
        """获取怪物 ID 并归一化"""
        monster_id = self.monster_db.get(monster_name, 0)
        return monster_id / self.max_monster_id
    
    def _get_potion_id(self, potion_name: str) -> float:
        """获取药水 ID 并归一化"""
        potion_id = self.potion_db.get(potion_name, 0)
        return potion_id / self.max_potion_id
    
    def _get_status_id(self, status_name: str) -> float:
        """获取 status ID 并归一化"""
        status_id = self.status_db.get(status_name, 0)
        return status_id / self.max_status_id
    
    def _get_status_type(self, status_type: str) -> float:
        """获取 status 类型编码 (Buff=1, Debuff=2)"""
        return self.status_type_map.get(status_type, 0)
    
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
            # 从 event 对象中获取 options
            event_data = state.get('event', {})
            options = event_data.get('options', [])
            
            # 检查是否有至少 3 个选项，并且第 3 个选项是"卷轴箱"
            if len(options) >= 3 and options[2].get("title") == "卷轴箱":
                num_options = 2
            else:
                num_options = len(options) if options else 3

            if action < 20:
                return ("event_option", action % num_options, None)
            else:
                return ("advance_dialogue", 0, None)
        
        # 战斗中手牌选择（升级/移除/转化）
        elif state_type == 'hand_select':
            hand_select = state.get('hand_select', {})
            cards = hand_select.get('cards', [])
            can_confirm = hand_select.get('can_confirm', False)
            num_cards = len(cards)
            
            if action < 10:  # 0-9: 选择/取消选择牌
                if action < num_cards:
                    return ("combat_select_card", action, None)
                else:
                    return ("wait", 0, None)
            elif action == 10:  # 10: 确认选择
                if can_confirm:
                    return ("combat_confirm_selection", 0, None)
                else:
                    return ("wait", 0, None)
            else:  # 11+: 等待
                return ("wait", 0, None)
        
        # 事件中卡牌选择（升级/移除/转化）
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
            
            elif state_type == 'hand_select':
                # 战斗中手牌选择（升级/移除/转化）
                if action_type == "combat_select_card":
                    result = self.client.combat_select_card(param)
                    info['action_result'] = result
                    self.log_event(f"选择手牌：{param}")
                elif action_type == "combat_confirm_selection":
                    result = self.client.combat_confirm_selection()
                    info['action_result'] = result
                    self.log_event("确认手牌选择")
                # wait: 等待
                time.sleep(0.5)
            
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
            import traceback
            traceback.print_exc()
            print(f"[DEBUG] state_type: {state.get('state_type', 'unknown')}")
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
        state_type = 'unknown'  # 初始化用于 debug
        
        try:
            # ==================== 状态类型 (1 维) ====================
            state_type = state.get('state_type', 'unknown')
            state_type_code = self.state_type_map.get(state_type, 1)  # 默认 unknown=1
            obs.append(state_type_code / 17.0)  # 归一化 (÷17，最大状态码)
            
            # 玩家数据 (所有状态都有)
            player = state.get('player', {})
        
            # ==================== 玩家基础维度 (5 维) ====================
            hp = player.get('hp', 0)
            max_hp = player.get('max_hp', 1)
            obs.append(hp / max(max_hp, 1))                          # HP 比例
            obs.append(player.get('energy', 0) / self.max_energy)    # 能量归一化 (÷5)
            obs.append(player.get('block', 0) / self.max_block)      # 格挡归一化 (÷50)
            obs.append(player.get('gold', 0) / self.max_gold)        # 金币归一化 (÷100)
            obs.append(state.get('run', {}).get('floor', 0) / self.max_floor)  # 楼层归一化 (÷60)
            
            # ==================== 玩家 Status 维度 (10 维) ====================
            player_status_list = player.get('status', [])[:5]  # 只取前 5 个
            for j in range(5):
                if j < len(player_status_list):
                    status = player_status_list[j]
                    status_name = status.get('name', '')
                    status_type = status.get('type', 'Unknown')
                    obs.append(self._get_status_id(status_name))      # status ID 归一化
                    obs.append(self._get_status_type(status_type))    # 类型编码
                else:
                    obs.extend([0, 0])  # 空槽位填充 0
            
            # ==================== 手牌/可选卡牌/商店 (最多 211 维) ====================
            if state_type == 'card_select':
                screen_type = state.get('card_select', {}).get('screen_type', 'unknown')
                screen_type_code = self.card_select_type_map.get(screen_type, 0)
                obs.append(screen_type_code / 3.0)
                
                cards = state.get('card_select', {}).get('cards', [])
                for i in range(self.max_cards):
                    if i < len(cards):
                        card = cards[i]
                        card_id_norm = self._get_card_id(card.get('name', ''))
                        type_enc = {'Attack': 1, 'Skill': 2, 'Power': 3}.get(card.get('type', ''), 0)
                        cost = card.get('cost', 0)
                        if cost == 'X': cost = 0
                        try:
                            cost = int(cost)
                        except:
                            cost = 0
                        desc_number = self._extract_desc_number(card.get('description', ''))
                        can_play = 1
                        is_upgraded = 1 if card.get('is_upgraded', False) else 0
                        obs.extend([card_id_norm, type_enc, cost, desc_number, can_play, is_upgraded])
                    else:
                        obs.extend([0, 0, 0, 0, 0, 0])
            
            elif state_type == 'shop':
                obs.append(0)
                items = state.get('shop', {}).get('items', [])
                for i in range(self.max_cards):
                    if i < len(items):
                        item = items[i]
                        category = item.get('category', 'unknown')
                        item_type = self.shop_item_type_map.get(category, 0)
                        obs.append(item_type / 4.0)
                        
                        if category == 'card':
                            item_id = self._get_card_id(item.get('card_name', ''))
                        elif category == 'relic':
                            item_id = self._get_relic_id(item.get('relic_name', ''))
                        elif category == 'potion':
                            item_id = self._get_potion_id(item.get('potion_name', ''))
                        else:
                            item_id = 0
                        obs.append(item_id)
                        obs.append(min(item.get('cost', 0) / 300.0, 1.0))
                        
                        if category == 'card':
                            desc_number = self._extract_desc_number(item.get('card_description', ''))
                        elif category == 'relic':
                            desc_number = self._extract_desc_number(item.get('relic_description', ''))
                        elif category == 'potion':
                            desc_number = self._extract_desc_number(item.get('potion_description', ''))
                        else:
                            desc_number = 0
                        obs.append(desc_number)
                        obs.append(1 if item.get('can_afford', False) else 0)
                        is_stocked = 1 if item.get('is_stocked', False) else 0
                        on_sale = 1 if item.get('on_sale', False) else 0
                        obs.append(is_stocked if not on_sale else 1)
                    else:
                        obs.extend([0, 0, 0, 0, 0, 0])
            
            else:
                obs.append(0)
                cards = []
                if state_type == 'card_reward':
                    cards = state.get('card_reward', {}).get('cards', [])
                elif state_type == 'hand_select':
                    # 战斗中手牌选择：读取 hand_select.cards
                    cards = state.get('hand_select', {}).get('cards', [])
                else:
                    cards = player.get('hand', [])
                
                for i in range(self.max_cards):
                    if i < len(cards):
                        card = cards[i]
                        card_id_norm = self._get_card_id(card.get('name', ''))
                        type_enc = {'Attack': 1, 'Skill': 2, 'Power': 3}.get(card.get('type', ''), 0)
                        cost = card.get('cost', 0)
                        if cost == 'X': cost = 0
                        try:
                            cost = int(cost)
                        except:
                            cost = 0
                        desc_number = self._extract_desc_number(card.get('description', ''))
                        can_play = 1 if card.get('can_play', True) else 0
                        is_upgraded = 1 if card.get('is_upgraded', False) else 0
                        obs.extend([card_id_norm, type_enc, cost, desc_number, can_play, is_upgraded])
                    else:
                        obs.extend([0, 0, 0, 0, 0, 0])
            
            # ==================== 敌人/事件状态 (170 维) ====================
            if state_type == 'event':
                event_data = state.get('event', {})
                event_name = event_data.get('event_name', '')
                obs.append(self._get_event_id(event_name))
                obs.append(1 if event_data.get('is_ancient', False) else 0)
                obs.append(1 if event_data.get('in_dialogue', False) else 0)
                
                # 检测是否有"卷轴箱"选项（通常是第 3 个），有的话只取前 2 个有效选项
                raw_options = event_data.get('options', [])
                if len(raw_options) >= 3 and raw_options[2].get("title") == "卷轴箱":
                    options = raw_options[:2]  # 排除卷轴箱
                else:
                    options = raw_options[:5]
                
                # 编码选项 (固定 5 个槽位，不足的用 0 填充)
                for j in range(5):
                    if j < len(options):
                        opt = options[j]
                        obs.append(opt.get('index', 0) / 10.0)
                        obs.append(self._extract_desc_number(opt.get('description', '')))
                        obs.append(1 if opt.get('is_locked', False) else 0)
                        obs.append(1 if opt.get('is_proceed', False) else 0)
                        obs.append(1 if opt.get('was_chosen', False) else 0)
                    else:
                        obs.extend([0, 0, 0, 0, 0])
                
                # 固定填充到 170 维 (事件 28 维 + 敌人空槽 142 维)
                obs.extend([0] * 142)
            else:
                enemies = state.get('battle', {}).get('enemies', [])
                for i in range(self.max_enemies):
                    if i < len(enemies):
                        enemy = enemies[i]
                        monster_id_norm = self._get_monster_id(enemy.get('name', ''))
                        obs.append(monster_id_norm)
                        obs.append(enemy.get('hp', 0) / max(enemy.get('max_hp', 1), 1))
                        type1, damage1, type2, damage2 = self._get_enemy_intents(enemy)
                        obs.append(type1 / 5.0)
                        obs.append(damage1 / self.max_intent_damage)
                        obs.append(type2 / 5.0)
                        obs.append(damage2 / self.max_intent_damage)
                        obs.append(min(len(enemy.get('status', [])), 5))
                        
                        status_list = enemy.get('status', [])[:5]
                        for j in range(5):
                            if j < len(status_list):
                                status = status_list[j]
                                status_name = status.get('name', '')
                                status_type = status.get('type', 'Unknown')
                                obs.append(self._get_status_id(status_name))
                                obs.append(self._get_status_type(status_type))
                            else:
                                obs.extend([0, 0])
                    else:
                        obs.extend([0] * 17)
            
            # 药水状态 (6 维)
            potions = player.get('potions', [])
            for i in range(6):
                if i < len(potions):
                    potion = potions[i]
                    potion_id_norm = self._get_potion_id(potion.get('name', ''))
                    obs.append(potion_id_norm)
                else:
                    obs.append(0)
        
        except Exception as e:
            print(f"\n[DEBUG] 观测编码错误：{e}")
            import traceback
            traceback.print_exc()
            print(f"[DEBUG] state_type: {state_type}")
            print(f"[DEBUG] obs 当前维度：{len(obs)}")
            print(f"[DEBUG] state 内容：{json.dumps(state, indent=2, ensure_ascii=False)[:2000]}")
            raise
        
        return np.array(obs, dtype=np.float32)
    
    def _extract_desc_number(self, description: str) -> float:
        """
        从卡牌描述中提取数字并归一化 (÷50)
        
        提取规则：
        - 提取描述中第一个数字
        - 如"造成 6 点伤害" → 6
        - 如"获得 12 点格挡" → 12
        - 无数字 → 0
        """
        import re
        if not description:
            return 0.0
        
        # 匹配第一个连续数字
        match = re.search(r'\d+', description)
        if match:
            number = int(match.group())
            return number / self.max_desc_number
        
        return 0.0
    
    def _get_enemy_intents(self, enemy: Dict) -> Tuple[float, float, float, float]:
        """
        解析敌人意图，返回 (意图类型 1, 攻击数值 1, 意图类型 2, 攻击数值 2)
        
        支持最多 2 个意图，不足的用 0 填充
        
        处理 label 格式：
        - "11" → 11
        - "2×4" → 2*4=8
        - 无值或非 Attack → 0
        """
        intents = enemy.get('intents', [])
        
        # 解析第一个意图
        type1, damage1 = 0.0, 0.0
        if len(intents) > 0:
            intent = intents[0]
            intent_type = intent.get('type', 'Unknown')
            type1 = self.intent_type_map.get(intent_type, 0)
            if intent_type == 'Attack':
                label = intent.get('label', '0')
                damage1 = float(self._parse_intent_label(label))
        
        # 解析第二个意图
        type2, damage2 = 0.0, 0.0
        if len(intents) > 1:
            intent = intents[1]
            intent_type = intent.get('type', 'Unknown')
            type2 = self.intent_type_map.get(intent_type, 0)
            if intent_type == 'Attack':
                label = intent.get('label', '0')
                damage2 = float(self._parse_intent_label(label))
        
        return (type1, damage1, type2, damage2)
    
    def _parse_intent_label(self, label: str) -> int:
        """
        解析意图 label 字符串，计算总伤害
        
        支持格式：
        - "11" → 11
        - "2×4" 或 "2*4" → 8
        - "1-3" → 2 (取平均)
        """
        if not label:
            return 0
        
        # 处理乘法格式 "2×4" 或 "2*4"
        if '×' in label:
            parts = label.split('×')
            try:
                result = 1
                for p in parts:
                    result *= int(p.strip())
                return result
            except:
                return 0
        
        if '*' in label:
            parts = label.split('*')
            try:
                result = 1
                for p in parts:
                    result *= int(p.strip())
                return result
            except:
                return 0
        
        # 处理范围格式 "1-3"
        if '-' in label:
            parts = label.split('-')
            try:
                low = int(parts[0].strip())
                high = int(parts[1].strip())
                return (low + high) // 2  # 取平均值
            except:
                return 0
        
        # 普通数字
        try:
            return int(label)
        except:
            return 0
    
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
    env = STS2Env()
    
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

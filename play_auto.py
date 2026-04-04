#!/usr/bin/env python3
"""
STS2 自动游玩脚本

使用训练好的 PPO 模型自动玩游戏，或者使用规则基策略。

使用方法:
    
    # 使用训练好的模型
    python play_auto.py --model ./models/sts2_ppo_20260403_120000/ppo_sts2_final.zip
"""

import argparse
import time
import sys
import numpy as np
from env.sts2_api import STS2Client, print_state_summary
from sts2_env import STS2Env


class RuleBasedAgent:
    """
    规则基智能体
    
    使用启发式规则做出决策，作为 RL 训练的基线。
    """
    
    def __init__(self):
        self.name = "RuleBased"
    
    def decide(self, state: dict, env: STS2Env) -> tuple:
        """
        根据当前状态做出决策
        
        Returns:
            (action, description) - 动作和描述
        """
        state_type = state.get('state_type', '')
        
        # ==================== 战斗状态 ====================
        if state_type in ['monster', 'elite', 'boss']:
            return self._combat_decision(state, env)
        
        # ==================== 奖励状态 ====================
        elif state_type == 'combat_rewards':
            return self._rewards_decision(state, env)
        
        # ==================== 卡牌奖励 ====================
        elif state_type == 'card_reward':
            return self._card_reward_decision(state, env)
        
        # ==================== 地图 ====================
        elif state_type == 'map':
            return self._map_decision(state, env)
        
        # ==================== 休息点 ====================
        elif state_type == 'rest_site':
            return self._rest_decision(state, env)
        
        # ==================== 商店 ====================
        elif state_type == 'shop':
            return self._shop_decision(state, env)
        
        # ==================== 事件 ====================
        elif state_type == 'event':
            return self._event_decision(state, env)
        
        # ==================== 未知状态 ====================
        else:
            return np.array([3, 0]), "跳过/等待"
    
    def _combat_decision(self, state: dict, env: STS2Env) -> tuple:
        """战斗决策"""
        battle = state.get('battle', {})
        player = battle.get('player', {})
        enemies = battle.get('enemies', [])
        
        if not player or not enemies:
            return np.array([3, 0]), "无效战斗状态"
        
        hand = player.get('hand', [])
        energy = player.get('energy', 0)
        
        # 1. 计算敌人威胁
        enemy_intent = 0
        for enemy in enemies:
            for intent in enemy.get('intents', []):
                if intent.get('type') == 'Attack':
                    try:
                        enemy_intent += int(intent.get('label', 0))
                    except:
                        pass
        
        # 3. 选择最佳卡牌
        best_card_idx = -1
        best_value = -1
        
        for i, card in enumerate(hand):
            if not card.get('can_play', False):
                continue
            
            cost = card.get('cost', 0)
            if cost == 'X':
                cost = 0
            try:
                cost = int(cost)
            except:
                cost = 0
            
            if cost > energy:
                continue
            
            # 计算卡牌价值
            value = self._evaluate_card(card, enemies, player)
            
            if value > best_value:
                best_value = value
                best_card_idx = i
        
        # 4. 决策
        if best_card_idx >= 0:
            card = hand[best_card_idx]
            return np.array([0, best_card_idx]), f"出牌：{card.get('name')} (价值：{best_value:.1f})"
        else:
            # 没有可出的牌，结束回合
            return np.array([2, 0]), "结束回合"
    
    def _evaluate_card(self, card: dict, enemies: list, player: dict) -> float:
        """评估卡牌价值"""
        value = 0
        card_type = card.get('type', '')
        desc = card.get('description', '')
        name = card.get('name', '')
        
        # 攻击牌
        if card_type == 'Attack':
            # 基础伤害
            if '造成' in desc:
                import re
                match = re.search(r'造成 (\d+) 点伤害', desc)
                if match:
                    damage = int(match.group(1))
                    value += damage * 1.5
            
            # 易伤/虚弱
            if '易伤' in desc:
                value += 5
            if '虚弱' in desc:
                value += 5
            
            # 多目标
            if '所有敌人' in desc or '全体' in desc:
                value *= len(enemies)
            
            # 补刀
            for enemy in enemies:
                hp = enemy.get('hp', 0)
                if '造成' in desc:
                    import re
                    match = re.search(r'造成 (\d+) 点伤害', desc)
                    if match and int(match.group(1)) >= hp:
                        value += 3
        
        # 防御牌
        elif card_type == 'Skill':
            # 格挡
            if '格挡' in desc:
                import re
                match = re.search(r'获得 (\d+) 点格挡', desc)
                if match:
                    block = int(match.group(1))
                    value += block * 1.2
            
            # 抽牌
            if '抽' in desc or '抓' in desc:
                value += 4
            
            # 能量
            if '能量' in desc:
                value += 5
        
        # 费用效率
        cost = card.get('cost', 0)
        if cost == 'X':
            cost = 0
        try:
            cost = int(cost)
        except:
            cost = 0
        
        if cost > 0:
            value /= cost
        
        return value
    
    def _rewards_decision(self, state: dict, env: STS2Env) -> tuple:
        """奖励选择决策"""
        rewards = state.get('rewards', [])
        
        if not rewards:
            return np.array([1, 0]), "无奖励"
        
        # 优先级：卡牌 > 药水 > 金币 > 其他
        for i, reward in enumerate(rewards):
            rtype = reward.get('type', '')
            if rtype in ['card', 'special_card']:
                return np.array([0, i]), f"领取卡牌奖励"
            elif rtype == 'potion':
                return np.array([0, i]), f"领取药水"
        
        # 默认选第一个
        return np.array([0, 0]), f"领取奖励：{rewards[0].get('type')}"
    
    def _card_reward_decision(self, state: dict, env: STS2Env) -> tuple:
        """卡牌奖励选择"""
        cards = state.get('cards', [])
        
        if not cards:
            return np.array([1, 0]), "跳过奖励"
        
        # 选择稀有度最高的
        rarity_order = {'Rare': 3, 'Uncommon': 2, 'Common': 1, 'Special': 4}
        best_idx = 0
        best_rarity = 0
        
        for i, card in enumerate(cards):
            rarity = rarity_order.get(card.get('rarity', 'Common'), 0)
            if rarity > best_rarity:
                best_rarity = rarity
                best_idx = i
        
        return np.array([0, best_idx]), f"选择卡牌：{cards[best_idx].get('name')}"
    
    def _map_decision(self, state: dict, env: STS2Env) -> tuple:
        """地图导航决策"""
        options = state.get('next_options', [])
        
        if not options:
            return np.array([1, 0]), "无路径"
        
        # 优先级：休息点 > 商店 > 普通战斗 > 宝藏 > 事件 > 精英 > Boss
        priority = {
            'RestSite': 1,
            'Shop': 2,
            'Monster': 3,
            'Treasure': 4,
            'Event': 5,
            'Ancient': 6,
            'Elite': 7,
            'Boss': 8,
        }
        
        best_idx = 0
        best_priority = 999
        
        for i, opt in enumerate(options):
            node_type = opt.get('node_type', 'Unknown')
            p = priority.get(node_type, 999)
            if p < best_priority:
                best_priority = p
                best_idx = i
        
        return np.array([0, best_idx]), f"前往：{options[best_idx].get('node_type')}"
    
    def _rest_decision(self, state: dict, env: STS2Env) -> tuple:
        """休息点决策"""
        options = state.get('options', [])
        player = state.get('battle', {}).get('player', {})
        
        if not options or not player:
            return np.array([1, 0]), "离开休息点"
        
        hp = player.get('hp', 0)
        max_hp = player.get('max_hp', 1)
        hp_percent = hp / max_hp if max_hp > 0 else 1
        
        # 血量低时休息，否则升级
        for i, opt in enumerate(options):
            opt_id = opt.get('id', '').lower()
            if hp_percent < 0.7 and 'rest' in opt_id:
                return np.array([0, i]), "休息（恢复 HP）"
            elif 'smith' in opt_id:
                return np.array([0, i]), "锻造（升级卡牌）"
        
        # 默认选第一个
        return np.array([0, 0]), f"选择：{options[0].get('name')}"
    
    def _shop_decision(self, state: dict, env: STS2Env) -> tuple:
        """商店决策"""
        # 简单策略：直接离开
        return np.array([1, 0]), "离开商店"
    
    def _event_decision(self, state: dict, env: STS2Env) -> tuple:
        """事件决策"""
        options = state.get('options', [])
        
        if not options:
            return np.array([1, 0]), "推进对话"
        
        # 随机选择一个可用选项
        available = [i for i, opt in enumerate(options) if not opt.get('locked', False)]
        if available:
            import random
            idx = random.choice(available)
            return np.array([0, idx]), f"选择：{options[idx].get('title')}"
        
        return np.array([1, 0]), "推进对话"


def play_with_model(model_path: str, num_episodes: int = 1):
    """使用训练好的模型游玩"""
    from stable_baselines3 import PPO
    
    print(f"\n🤖 加载模型：{model_path}")
    env = STS2Env()
    model = PPO.load(model_path, env=env)
    
    for ep in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"🎮 第 {ep+1} 局")
        print(f"{'='*60}")
        
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1
            
            state = info.get('state', {})
            state_type = state.get('state_type', '?')
            print(f"Step {step}: {state_type}, reward={reward:.2f}, total={total_reward:.2f}")
            
            if step > 500:  # 防止无限循环
                break
        
        print(f"\n✅ 局结束 - 总步数：{step}, 总奖励：{total_reward:.2f}")
    
    env.close()


def play_with_rules(num_episodes: int = 1):
    """使用规则基策略游玩"""
    print("\n🧠 使用规则基策略\n")
    
    agent = RuleBasedAgent()
    client = STS2Client()
    
    for ep in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"🎮 第 {ep+1} 局")
        print(f"{'='*60}\n")
        
        # 等待游戏开始
        print("等待游戏开始...")
        while True:
            try:
                state = client.get_game_state()
                if state.get('state_type') not in ['menu', 'unknown']:
                    break
            except:
                pass
            time.sleep(1)
        
        step = 0
        max_steps = 1000
        
        while step < max_steps:
            try:
                state = client.get_game_state()
                state_type = state.get('state_type', 'unknown')
                
                # 检查是否死亡
                hp = state.get('battle', {}).get('player', {}).get('hp', 0)
                if hp <= 0:
                    print("\n💀 游戏结束 - 角色死亡")
                    break
                
                # 创建临时环境用于决策
                env = STS2Env()
                action, description = agent.decide(state, env)
                env.close()
                
                print(f"[{step:3d}] {state_type:15s} → {description}")
                
                # 执行动作
                action_type = int(action[0])
                action_param = int(action[1])
                
                if state_type in ['monster', 'elite', 'boss']:
                    if action_type == 0:  # 出牌
                        client.combat_play_card(action_param)
                    elif action_type == 1:  # 用药
                        client.use_potion(action_param)
                    elif action_type == 2:  # 结束回合
                        client.combat_end_turn()
                elif state_type == 'combat_rewards':
                    if action_type == 0:
                        client.rewards_claim(action_param)
                elif state_type == 'card_reward':
                    if action_type == 0:
                        client.rewards_pick_card(action_param)
                    else:
                        client.rewards_skip_card()
                elif state_type == 'map':
                    if action_type == 0:
                        client.map_choose_node(action_param)
                elif state_type == 'rest_site':
                    if action_type == 0:
                        client.rest_choose_option(action_param)
                    else:
                        client.proceed_to_map()
                elif state_type == 'shop':
                    if action_type == 1:
                        client.proceed_to_map()
                elif state_type == 'event':
                    if action_type == 0:
                        client.event_choose_option(action_param)
                    else:
                        client.event_advance_dialogue()
                
                step += 1
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                print("\n👋 手动停止")
                break
            except Exception as e:
                print(f"❌ 错误：{e}")
                time.sleep(1)
        
        print(f"\n✅ 局结束 - 总步数：{step}")


def main():
    parser = argparse.ArgumentParser(description="STS2 自动游玩")
    parser.add_argument("--model", type=str, default=None, help="PPO 模型路径")
    parser.add_argument("--episodes", type=int, default=1, help="游玩局数")
    parser.add_argument("--rules", action="store_true", help="使用规则基策略")
    
    args = parser.parse_args()
    
    if args.model:
        play_with_model(args.model, args.episodes)
    else:
        play_with_rules(args.episodes)


if __name__ == "__main__":
    main()

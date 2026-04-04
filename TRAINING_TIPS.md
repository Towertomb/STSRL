# STS2 PPO 训练调优指南

## 🐛 问题：模型只会结束回合

### 原因分析

1. **奖励函数偏差**
   - 结束回合 = 无风险、无惩罚
   - 出牌 = 可能出错、需要目标选择
   - 模型学会"不作为 = 安全"

2. **探索不足**
   - 熵系数太低 (0.01)
   - 模型过早收敛到局部最优

3. **动作空间设计**
   - 结束回合总是可用
   - 出牌需要条件（能量、目标）

---

## ✅ 已应用的修复

### 1. 奖励函数调整

```python
# 之前
reward += damage_dealt * 0.1  # 伤害奖励
reward += killed * 10.0       # 击杀奖励

# 现在
reward += damage_dealt * 0.2  # 翻倍
reward += killed * 15.0       # 增加 50%
reward += cards_played * 1.0  # 新增：出牌奖励
reward -= 0.5                 # 新增：有能量却结束回合的惩罚
```

### 2. 增加探索

```python
# 之前
ent_coef = 0.01

# 现在
ent_coef = 0.03  # 3 倍探索
```

---

## 📊 训练建议

### 重新训练（必须）

```powershell
cd D:\STSRL

# 删除旧模型（避免加载旧策略）
rm -r ./models

# 开始新训练
python train_ppo.py --total-timesteps 500000 --ent-coef 0.03
```

### 监控指标

使用 TensorBoard 查看：
```powershell
tensorboard --logdir ./logs
```

**关键指标：**
| 指标 | 期望趋势 | 说明 |
|------|---------|------|
| `rollout/ep_rew_mean` | 上升 | 平均奖励增加 |
| `rollout/ep_len_mean` | 上升 | 战斗持续时间增加 |
| `loss/value_loss` | 下降 | 价值预测准确 |
| `policy/entropy` | 缓慢下降 | 探索减少（正常） |

---

## 🔧 进阶调优

### 如果还是只结束回合...

#### 方案 A：更大的出牌奖励
```python
# 在 sts2_env.py 中
reward += cards_played * 2.0  # 从 1.0 提高到 2.0
```

#### 方案 B：增加训练步数
```powershell
python train_ppo.py --total-timesteps 1000000
```

#### 方案 C：修改动作空间
让结束回合在有能量时不可用：
```python
# 在 step() 中
if action_type == 2 and energy > 0:
    reward -= 1.0  # 更大的惩罚
    # 或者强制改为出牌动作
```

#### 方案 D：课程学习
1. 先训练简单场景（只有出牌，不能结束回合）
2. 逐步增加复杂度

---

## 📈 预期训练曲线

```
奖励
  ↑
  |         ╱────
  |       ╱
  |     ╱
  |   ╱
  | ╱
  └────────────→ 步数
  
初期 (0-50k): 奖励波动大，模型探索
中期 (50k-200k): 开始学会出牌
后期 (200k+): 策略稳定，奖励上升
```

---

## 🎯 快速验证

训练 10k 步后，测试模型：
```powershell
python play_auto.py --model ./models/sts2_ppo_xxx/ppo_sts2_final.zip
```

**观察：**
- 是否会出牌了？
- 出牌选择是否合理？
- 还是只结束回合？

根据结果调整奖励参数。

---

## 💡 总结

| 问题 | 解决方案 |
|------|---------|
| 只结束回合 | 增加出牌奖励 + 结束回合惩罚 |
| 探索不足 | 提高熵系数到 0.03 |
| 训练慢 | 增加伤害/击杀奖励权重 |
| 策略不稳定 | 增加训练步数 |

**核心原则：** 让模型从"出牌"获得的奖励 > "结束回合"的奖励

# STS2 RL - 杀戮尖塔 2 强化学习试验

使用 PPO 算法训练 AI 自动玩《杀戮尖塔 2》(Slay the Spire 2)。

---

## 项目结构

```
D:\STSRL\
├── train_ppo.py              # PPO 训练脚本（带自动重开）
├── play_auto.py              # 自动游玩脚本（规则基策略）
├── check_state.py            # 状态检查工具
├── scripts/
│   ├── auto_restart.py       # 死亡自动重开模块
│   ├── clicker.py            # 图像识别点击器
│   └── UIimage/              # UI 截图目录
│       ├── ensue.png
│       ├── main_manu.png
│       ├── single_mode.png
│       ├── standard_mode.png
│       └── check_in.png
├── env/
│   ├── sts2_api.py           # STS2 MCP API 封装
│   └── sts2_env.py           # Gymnasium 环境包装器
├── logs/                     # 训练日志（TensorBoard）
├── models/                   # 保存的模型
└── requirements.txt          # Python 依赖
```

---

## 快速开始

### 1️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

### 2️⃣ 准备游戏

1. 确保 STS2_MCP mod 已安装到游戏目录
2. 启动《杀戮尖塔 2》
3. 启用 mods
4. 开始一局游戏

### 3️⃣ 测试连接

```bash
python check_state.py
```

### 4️⃣ 开始训练

```bash
python train_ppo.py --total-timesteps 1000000
```

---

## 核心功能

### ✅ PPO 训练

- 使用 Stable Baselines3 的 PPO 算法
- MLP 策略网络
- 支持断点续训

### ✅ 自动重开

训练过程中检测到死亡时自动：
1. 点击"继续"
2. 返回主菜单
3. 选择单人/标准模式
4. 开始新游戏
5. 等待涅奥事件（交给模型决策）

### ✅ 图像识别点击

基于 OpenCV 模板匹配的 UI 自动点击，用于重开流程。

---

## 训练监控

使用 TensorBoard 查看训练指标：

```bash
tensorboard --logdir ./logs
```

**关键指标：**
- `rollout/ep_rew_mean` - 平均奖励（应上升）
- `rollout/ep_len_mean` - 平均回合长度
- `loss/value_loss` - 价值损失（应下降）
- `policy/entropy` - 策略熵（缓慢下降为正常）

---

## 配置选项

### 训练参数

```bash
python train_ppo.py \
  --total-timesteps 1000000 \
  --learning-rate 0.0003 \
  --ent-coef 0.03 \
  --save-freq 10000
```

### 自动重开配置 (角色死亡时会自动重开)

在 `train_ppo.py` 中修改：

```python
auto_restart = AutoRestarter(
    check_interval=2.0,   # 状态检测间隔（秒）
    confidence=0.9        # 图像识别置信度
)
```

---

## 工具脚本

| 脚本 | 功能 |
|------|------|
| `check_state.py` | 测试游戏连接，打印当前状态 |
| `play_auto.py` | 使用规则基策略自动游玩 |
| `train_ppo.py` | PPO 训练（带自动重开） |

---



## 注意事项

1. **游戏必须运行** - 训练前确保 STS2 已启动且 mod 已启用
2. **窗口位置固定** - 避免移动游戏窗口（影响图像识别）
3. **UI 截图准确** - 确保 `UIimage/` 中的截图与游戏实际显示一致
4. **紧急停止** - 鼠标移到屏幕角落可触发 PyAutoGUI 安全机制

---

## 相关资源

- [STS2MCP Mod](https://github.com/kunology/STS2MCP) - 游戏 API mod
- [Stable Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [Gymnasium 文档](https://gymnasium.farama.org/)

---

## 训练目标 (还没成功训练出来)

让模型学会：
- ✅ 合理出牌（而非只结束回合）
- ✅ 正确使用药水
- ✅ 选择地图路径
- ✅ 休息点决策
- ✅ 奖励选择

---

## 计划

- 优化观测空间，增加维度 （敌人id，手牌id...）
- 优化模型动作输出，考虑更多情况 ）
- 增加奖励函数，考虑更多游戏机制
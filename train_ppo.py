#!/usr/bin/env python3
"""
STS2 PPO 训练脚本

使用 Stable Baselines3 的 PPO 算法训练 STS2 智能体。

使用方法:
    python train_ppo.py --total-timesteps 1000000 --save-freq 10000
"""

import argparse
import os
import sys
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from env.sts2_env import STS2Env as STS2Env
from scripts.auto_restart import AutoRestarter

# 创建全局自动重开实例
auto_restart = AutoRestarter(check_interval=2.0, confidence=0.9)


def make_env(log_dir: str, rank: int = 0) -> gym.Env:
    """创建带监控的环境"""
    def _init():
        env = STS2Env()
        env = Monitor(env, os.path.join(log_dir, f"monitor_{rank}"))
        return env
    return _init


def train(
    total_timesteps: int = 100000,
    log_dir: str = "./logs",
    save_dir: str = "./models",
    save_freq: int = 10000,
    eval_freq: int = 5000,
    n_eval_episodes: int = 3,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    verbose: int = 1,
    resume_from: str = None,
):
    """
    训练 PPO 智能体
    
    Args:
        total_timesteps: 总训练步数
        log_dir: 日志目录
        save_dir: 模型保存目录
        save_freq: 保存频率（步数）
        eval_freq: 评估频率（步数）
        n_eval_episodes: 评估回合数
        learning_rate: 学习率
        n_steps: PPO 每更新的步数
        batch_size: 批次大小
        n_epochs: PPO 每更新的 epoch 数
        gamma: 折扣因子
        gae_lambda: GAE lambda
        clip_range: PPO clip 范围
        ent_coef: 熵系数
        verbose: 详细程度
        resume_from: 从哪个模型恢复训练
    """
    
    # 创建目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sts2_ppo_{timestamp}"
    current_log_dir = os.path.join(log_dir, run_name)
    current_save_dir = os.path.join(save_dir, run_name)
    
    os.makedirs(current_log_dir, exist_ok=True)
    os.makedirs(current_save_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("🎮 STS2 PPO 训练")
    print("=" * 60)
    print(f"\n📁 日志目录：{current_log_dir}")
    print(f"💾 保存目录：{current_save_dir}")
    print(f"📊 总步数：{total_timesteps:,}")
    print(f"📈 学习率：{learning_rate}")
    print("=" * 60 + "\n")
    
    # 创建环境
    print("🔧 创建环境...")
    env = DummyVecEnv([make_env(current_log_dir)])
    
    # 环境归一化（可选，有助于训练稳定性）
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # 创建或加载模型
    if resume_from:
        print(f"📥 从 {resume_from} 恢复训练...")
        model = PPO.load(resume_from, env=env)
    else:
        print("🧠 创建新 PPO 模型...")
        model = PPO(
            policy="MlpPolicy",  # 使用 MLP 策略，因为观测空间是 Box 向量
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=verbose,
            tensorboard_log=current_log_dir,
        )
    
    # 回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=current_save_dir,
        name_prefix="ppo_sts2",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # ============ 启动自动重开 ============
    auto_restart.start()
    # ====================================
    
    # 开始训练
    print("\n🚀 开始训练...\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            tb_log_name=run_name,
        )
        
        # 保存最终模型
        final_path = os.path.join(current_save_dir, "ppo_sts2_final")
        model.save(final_path)
        print(f"\n✅ 训练完成！模型已保存到：{final_path}")
        
    except KeyboardInterrupt:
        print("\n\n👋 训练被中断")
        interrupted_path = os.path.join(current_save_dir, "ppo_sts2_interrupted")
        model.save(interrupted_path)
        print(f"💾 已保存中断时的模型：{interrupted_path}")
    
    finally:
        # ============ 停止自动重开 ============
        auto_restart.stop()
        # ====================================
    
    env.close()
    return model


def main():
    parser = argparse.ArgumentParser(description="STS2 PPO 训练")
    parser.add_argument("--total-timesteps", type=int, default=100000, help="总训练步数")
    parser.add_argument("--log-dir", type=str, default="./logs", help="日志目录")
    parser.add_argument("--save-dir", type=str, default="./models", help="模型保存目录")
    parser.add_argument("--save-freq", type=int, default=10000, help="保存频率")
    parser.add_argument("--eval-freq", type=int, default=5000, help="评估频率")
    parser.add_argument("--learning-rate", type=float, default=2e-3, help="学习率")
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO n_steps")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--ent-coef", type=float, default=0.03, help="熵系数（增加探索）")
    parser.add_argument("--resume-from", type=str, default=None, help="恢复训练的模型路径")
    parser.add_argument("--verbose", type=int, default=1, help="详细程度")
    
    args = parser.parse_args()
    
    train(
        total_timesteps=args.total_timesteps,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        verbose=args.verbose,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()

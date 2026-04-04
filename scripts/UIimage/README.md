# STS2 UI 图像截图指南

将以下 UI 元素的截图保存到此目录，用于自动重开功能。

## 📸 需要截图的 UI 元素

| 文件名 | 描述 | 截图时机 |
|--------|------|---------|
| `ensue.png` | 游戏结束后的"继续"按钮 | 死亡后出现的界面 |
| `main_manu.png` | "主菜单"按钮 | 游戏结束界面 |
| `single_mode.png` | "单人模式"按钮 | 主菜单 |
| `standard_mode.png` | "标准模式"按钮 | 模式选择界面 |
| `check_in.png` | "签到"或"开始游戏"按钮 | 每日签到或游戏开始界面 |

## 📷 截图要求

1. **格式**: PNG
2. **清晰度**: 确保按钮完整清晰
3. **大小**: 不要太大，包含按钮本身即可（约 100x50 像素）
4. **背景**: 尽量与游戏中实际显示的背景一致

## 🛠️ 截图方法

### Windows 自带截图
1. 按 `Win + Shift + S`
2. 选择要截图的按钮区域
3. 保存为 PNG 文件

### 或使用 Python 截图
```python
import pyautogui

# 截图整个屏幕
screenshot = pyautogui.screenshot()
screenshot.save("screenshot.png")

# 或截图指定区域
screenshot = pyautogui.screenshot(region=(x, y, width, height))
screenshot.save("button.png")
```

## ✅ 验证截图

运行测试脚本验证截图是否可用：

```powershell
cd D:\STSRL
python scripts\auto_restart.py
```

## 📁 目录结构

```
D:\STSRL\
├── scripts/
│   ├── UIimage/
│   │   ├── ensue.png       ← 放这里
│   │   ├── main_manu.png   ← 放这里
│   │   ├── single_mode.png ← 放这里
│   │   ├── standard_mode.png ← 放这里
│   │   └── check_in.png    ← 放这里
│   ├── auto_restart.py
│   └── clicker.py
└── train_ppo.py
```

# DemoGen 机器人仿真项目

本项目整合了 [DemoGen](https://github.com/TEA-Lab/DemoGen) 合成演示数据生成框架与 [RoboPal](https://github.com/NoneJou072/robopal) 机器人仿真环境，用于机器人操作的视觉运动策略学习研究。

## 项目概述

本项目基于以下核心组件：
- **DemoGen**: 合成演示数据生成方法，仅凭一个真实世界的人类演示即可在几秒内生成数百个空间增强的合成演示
- **RoboPal**: 基于 MuJoCo 物理引擎的多平台模块化机器人仿真框架
- **Diffusion Policy**: 3D 扩散策略实现，用于训练视觉运动策略

## 目录结构

```
demogen/
├── DemoGen-master/          # DemoGen 主项目
│   ├── demo_generation/     # 演示生成核心代码
│   ├── diffusion_policies/  # 扩散策略实现
│   ├── data/                # 数据集目录
├── robopal/                 # RoboPal 机器人仿真环境（适配 Python 3.8）
│   ├── robopal/             # 核心代码
│   ├── assets/             # 机器人模型和场景
│   └── demos/              # 演示脚本
└── README.md               # 本文件
```

## 环境配置

### 1. 系统要求

- **操作系统**: Linux (推荐 Ubuntu 20.04+)
- **Python**: 3.8
- **Conda**: 用于环境管理

### 2. 创建 Conda 环境

```bash

# 创建新的 Python 3.8 环境
conda create -n demogen python=3.8
conda activate demogen
```

### 3. 安装基础依赖

```bash
# 安装 Python 基础包
pip3 install imageio imageio-ffmpeg termcolor hydra-core==1.2.0 \
    zarr==2.12.0 matplotlib setuptools==59.5.0 pynput h5py \
    scikit-video tqdm

# 安装其他科学计算包
pip install numpy scipy scikit-learn pillow
```

### 4. 安装 Diffusion Policy

```bash
cd DemoGen-master/diffusion_policies
pip install -e .
cd ../..
```

### 5. 安装 DemoGen

```bash
cd DemoGen-master/demo_generation
pip install -e .
cd ../..
```

### 6. 安装 RoboPal（已适配 Python 3.8）

RoboPal 已经过修改以兼容 Python 3.8，主要改动包括：
- 修复了 Python 3.8 兼容性问题
- 添加了 DemoGen 相机配置
- 实现了标准数据格式输出

```bash
cd robopal
pip install -r requirements.txt
pip install -e .
cd ..
```

### 7. 安装 MuJoCo

本项目使用 MuJoCo 物理引擎进行仿真：

```bash
pip install mujoco
```

### 8. 验证安装

运行 RoboPal 键盘遥操作演示验证安装：

```bash
python robopal/robopal/demos/demo_keyboard_teleop.py
```

**键盘控制说明**：
- 方向键 ↑/↓/←/→: 控制末端在 x/y 平面移动（自动采集）
- Ctrl + ↑/↓: 控制 z 轴移动（自动采集）
- CapsLock: 切换爪子开/闭状态（自动采集）
- 空格键: 手动录制当前状态一帧
- r 键: 重置环境
- q 键: 保存轨迹为 pickle 文件
- c 键: 清除当前轨迹
- ESC: 退出

## 数据格式

本项目使用标准绝对位置格式，采集的数据包含：

- **point_cloud**: (T, 1024, 6) - XYZ+RGB 颜色
- **image**: (T, 3, 84, 84) - RGB CHW 格式
- **depth**: (T, 84, 84) - 深度图
- **agent_pos**: (T, 7) - [x,y,z,qx,qy,qz,gripper] 当前实际位置
- **action**: (T, 7) - [x,y,z,qx,qy,qz,gripper] 目标位置

## 快速开始

本项目提供完整的从数据收集到策略训练和部署的工作流程：

### 步骤 1: 收集仿真数据

使用键盘遥操作 Panda 机械臂收集演示数据：

```bash
python robopal/robopal/demos/demo_keyboard_teleop.py
```

**操作说明**：
- 方向键控制末端移动，CapsLock 控制夹爪
- 程序会自动采集每帧数据（点云、RGB图像、深度图、状态、动作）
- 按 `q` 键保存轨迹到 `robopal/robopal/demos/collected_data/`
- 按 `ESC` 退出

### 步骤 2: 格式转换为 Zarr

将收集的 pickle 数据转换为 Zarr 格式供 DemoGen 使用：

```bash
cd DemoGen-master
python merge_zarr.py <实验名称>
```

数据将从 `data/source_demos/<实验名称>/` 读取并保存到 `data/datasets/source/<实验名称>.zarr`

### 步骤 3: 图像分割（SAM）

使用 Segment Anything Model (SAM) 对目标物体进行分割：

```bash
cd DemoGen-master/data/sam_mask
python segment_interactive.py \
    --image "0216-cube/source.jpg" \
    --output "0216-cube/green cube.jpg"
```

**操作说明**：
- 左键点击选择前景点（绿色）
- 右键点击选择背景点（红色）
- 按 `s` 执行分割，按 `q` 保存并退出

### 步骤 4: 生成合成数据

使用 DemoGen 生成大量合成演示数据：

```bash
cd DemoGen-master/demo_generation
python gen_demo.py --config-path config/<配置文件>.yaml
```

或使用提供的脚本：
```bash
bash run_gen_demo.sh
```

生成的数据将保存在 `data/datasets/generated/` 目录

### 步骤 5: 训练策略模型

使用 DP3 (3D Diffusion Policy) 训练视觉运动策略：

```bash
cd DemoGen-master/diffusion_policies
bash train.sh <实验名称> dp3 <任务名称> <随机种子>
```

示例：
```bash
bash train.sh 0216-cube_test_100 dp3 cube 0
```

训练过程中会在终端显示实时进度，模型检查点保存在 `data/ckpts/` 目录

### 步骤 6: 部署和评估

在仿真环境中部署训练好的策略：

```bash
cd DemoGen-master/diffusion_policies
python evaluate_sim.py checkpoint=../data/ckpts/<检查点路径>/checkpoints/<模型编号>.ckpt
```

### （可选）步骤 7: 回放训练数据

验证训练数据质量，直接执行训练数据中记录的动作：

```bash
cd DemoGen-master/diffusion_policies
python replay_training_data.py
```

## 工作流程图

```
键盘遥操作收集 → Zarr格式转换 → SAM图像分割 → DemoGen生成合成数据
                                                  ↓
                                           DP3策略训练
                                                  ↓
                                           仿真环境部署评估
```

## 引用

如果本项目对您的研究有帮助，请引用以下论文：

### DemoGen
```bibtex
@article{xue2025demogen,
  title={DemoGen: Synthetic Demonstration Generation for Data-Efficient Visuomotor Policy Learning},
  author={Xue, Zhengrong and Deng, Shuying and Chen, Zhenyang and Wang, Yixuan and Yuan, Zhecheng and Xu, Huazhe},
  journal={arXiv preprint arXiv:2502.16932},
  year={2025}
}
```

### RoboPal
```bibtex
@software{Zhou_robopal_A_Simulation_2024,
author = {Zhou, Haoran and Huang, Yichao and Zhao, Yuhan and Lu, Yang},
doi = {10.5281/zenodo.11078757},
month = apr,
title = {{robopal: A Simulation Framework based Mujoco}},
url = {https://github.com/NoneJou072/robopal},
version = {0.3.1},
year = {2024}
}
```

## 许可证

本项目遵循相关子项目的许可证：
- DemoGen: MIT License
- RoboPal: Apache 2.0 License 

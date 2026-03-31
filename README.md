# 🦖 用RL玩谷歌小恐龙游戏

🎮 本项目使用强化学习算法（D3QN + PER）控制浏览器中的谷歌小恐龙游戏。

## 📁 1. 游戏下载与文件配置

1. 🔗 访问游戏源码地址：https://github.com/wayou/t-rex-runner
2. ⬇️ 点击 "Code" 选择 "Download ZIP" 下载源码包。
3. 📦 解压源码包，将提取出的 `index.html` 文件及配套资源放置于本项目的根目录下。
4. 📂 确保项目目录结构包含以下核心文件：
   - 📄 `index.html` (游戏页面)
   - 📄 `dino_env.py` (环境封装代码)
   - 📄 `train_d3qn.py` (训练代码)
   - 📄 `model_test.py` (测试代码)

## 🛠️ 2. 环境依赖

🐍 项目基于 Python 运行。请在终端执行以下命令安装依赖库：

```bash
pip install gymnasium numpy playwright torch opencv-python imageio matplotlib
```

🌐 依赖库安装完成后，执行以下命令安装 Playwright 所需的浏览器内核：

```bash
playwright install chromium
```

## 🧠 3. 算法设计与MDP定义

### 🏗️ 算法结构

⚙️ 项目采用 D3QN（Dueling Double DQN）算法结合 PER（优先经验回放）机制。

* ♊ **Double 结构：** 算法使用两个独立的神经网络。在线网络（Online Network）负责在给定状态下选择动作，目标网络（Target Network）负责计算该动作的 Q 值。此结构用于减缓 Q 值的过估计现象。
* ⚔️ **Dueling 结构：** 神经网络在全连接层之前产生分支。一个分支输出状态价值 V(s)，另一个分支输出各个动作的优势 A(s, a)。网络输出层的计算方式为 Q(s, a) = V(s) + A(s, a) - mean(A)。
* 🎯 **PER (优先经验回放)：** 经验回放池底层使用 SumTree 数据结构。系统计算每个样本的 TD 误差（TD Error），并依此计算优先级。TD 误差数值越大的样本，在训练中被采样的概率越高。

### 📊 MDP (马尔可夫决策过程) 要素

* 🖼️ **状态空间 (State Space)：** 形状为 (4, 84, 84) 的矩阵。数据来源于浏览器画布截图，依次经过 RGB 转灰度、尺寸调整为 84x84 的处理。系统利用队列存储并堆叠时间序列上的连续 4 帧图像。
* 🕹️ **动作空间 (Action Space)：** 包含 3 个选项的离散一维空间。
    * `0`: 不操作 (跑)
    * `1`: 触发键盘空格键 (跳跃)
    * `2`: 触发键盘下方向键 (低头)
* 🏆 **奖励函数 (Reward Function)：** * ➕ **存活奖励**：环境步进时未触发结束条件，获得 0.1。
    * ➖ **碰撞惩罚**：环境读取到游戏结束标志，获得 -10.0。

## 🚀 4. 运行与测试步骤

### 🏋️ 训练模型

💻 在终端运行以下命令开始训练：

```bash
python train_d3qn.py
```

🤖 训练脚本将打开无头浏览器进行交互。模型权重参数文件（`.pth` 格式）会按设定的步数间隔存储于 `./dino_models` 文件夹中。

### 🎬 测试模型与保存记录

1. 📝 打开 `model_test.py` 文件。
2. 🔍 找到代码底部的 `MODEL_WEIGHTS` 变量，将路径字符串修改为需要测试的模型文件绝对路径或相对路径（例如 `./dino_models/d3qn_step_210000.pth`）。
3. 🏃 在终端运行以下命令：

```bash
python model_test.py
```

📺 测试脚本将弹出带有可视界面的浏览器，使用设定权重的网络模型输出动作。运行过程中，控制台输出实时步数、动作选择和得分。每局测试触碰障碍物结束后，代码会将彩色原图帧组装，并在 `./eval_gifs` 目录下生成帧率为 20 FPS 的 GIF 动画文件。

![ep1_steps275](https://github.com/user-attachments/assets/700ab15b-1bcc-4514-8a37-9d40074b5dea)

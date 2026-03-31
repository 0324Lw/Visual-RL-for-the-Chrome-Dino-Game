import os
import time
import torch
import torch.nn as nn
import numpy as np
import cv2
import imageio
from dino_env import DinoVisionEnv


# ==========================================
# 1. 重构网络结构 (与训练时严格对齐)
# ==========================================
def orthogonal_init(layer, gain=1.0):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    return layer


class DuelingCNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuelingCNN, self).__init__()
        self.conv = nn.Sequential(
            orthogonal_init(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), gain=np.sqrt(2)),
            nn.ReLU(),
            orthogonal_init(nn.Conv2d(32, 64, kernel_size=4, stride=2), gain=np.sqrt(2)),
            nn.ReLU(),
            orthogonal_init(nn.Conv2d(64, 64, kernel_size=3, stride=1), gain=np.sqrt(2)),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.advantage = nn.Sequential(
            orthogonal_init(nn.Linear(conv_out_size, 512), gain=np.sqrt(2)),
            nn.ReLU(),
            orthogonal_init(nn.Linear(512, num_actions), gain=1.0)
        )
        self.value = nn.Sequential(
            orthogonal_init(nn.Linear(conv_out_size, 512), gain=np.sqrt(2)),
            nn.ReLU(),
            orthogonal_init(nn.Linear(512, 1), gain=1.0)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.float() / 255.0
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        adv = self.advantage(features)
        val = self.value(features)
        return val + adv - adv.mean(dim=1, keepdim=True)


# ==========================================
# 2. 边渲染边录制的测试主函数
# ==========================================
def test_and_record_realtime(model_path, num_episodes=5, gif_fps=20):
    print("正在启动游戏环境 (带界面)...")
    env = DinoVisionEnv()

    # 测试评估使用 CPU 即可，速度完全够用
    device = torch.device("cpu")
    print(f"使用计算设备: {device}")

    # 初始化网络并加载模型
    model = DuelingCNN((4, 84, 84), 3).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型权重: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 设置为评估模式
    model.eval()

    # 动作文本映射与颜色字典 (BGR 格式)
    action_text = {0: "RUN", 1: "JUMP", 2: "DUCK"}
    action_color = {0: (0, 255, 0), 1: (255, 100, 100), 2: (0, 100, 255)}

    # 终端输出映射
    console_text = {0: "跑 (RUN)  ", 1: "跳跃 (JUMP)", 2: "低头 (DUCK)"}

    # 创建保存 GIF 的文件夹
    save_dir = "./eval_gifs"
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        step_count = 0
        episode_reward = 0.0
        frames_for_gif = []

        print(f"\n{'=' * 40}")
        print(f"开始测试第 {ep} 局游戏 | 准备录制...")
        print(f"{'=' * 40}")

        time.sleep(0.5)

        while not done:
            # 1. 模型前向传播，纯贪婪策略
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
                action = q_values.argmax().item()

            # 2. 与环境交互 (环境内部会等待 100ms 并更新画面)
            next_state, reward, done, _, _ = env.step(action)

            # ======== 【修改部分开始】 ========
            # 3. 直接从环境中读取刚才 step 时顺手保存的彩色帧，耗时 0 毫秒！
            # 注意：强制统一尺寸防报错的逻辑依然保留
            frame = cv2.resize(env.latest_color_frame, (600, 150))
            # ======== 【修改部分结束】 ========

            # 4. 绘制动作指令文本
            text = f"Action: {action_text[action]}"
            color = action_color[action]
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_for_gif.append(frame_rgb)

            # 5. 累计数据
            step_count += 1
            episode_reward += reward
            state = next_state

            print(
                f"步数: {step_count:04d} | 动作: {console_text[action]} | 步奖励: {reward:+.1f} | 累计得分: {episode_reward:+.1f}")

        print(f"--- 第 {ep} 局结束，小恐龙撞毁 ---")
        print(f"最终存活步数: {step_count} | 总得分: {episode_reward:.1f}")
        print(f"正在将 {step_count} 帧画面渲染为 GIF，请稍候...")

        # 6. 渲染 GIF (20 FPS 相当于 2倍速)
        gif_filename = os.path.join(save_dir, f"ep{ep}_steps{step_count}.gif")
        imageio.mimsave(gif_filename, frames_for_gif, fps=gif_fps)
        print(f"GIF 已保存: {gif_filename}")

        # 死亡后暂停一会
        time.sleep(1.5)

    env.close()
    print("\n所有测试局运行与录制完毕！")


if __name__ == "__main__":
    # 请替换为你跑出 600 步好成绩的那个权重文件
    MODEL_WEIGHTS = "./dino_models/d3qn_step_220000.pth"

    if not os.path.exists(MODEL_WEIGHTS):
        print(f"找不到模型文件: {MODEL_WEIGHTS}，请确认路径。")
    else:
        # fps=20 保证了长视频的播放节奏，既清楚又不会过慢
        test_and_record_realtime(MODEL_WEIGHTS, num_episodes=5, gif_fps=5)
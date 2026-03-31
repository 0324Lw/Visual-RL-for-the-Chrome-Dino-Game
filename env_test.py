from dino_env import DinoVisionEnv
import time
import cv2

if __name__ == '__main__':
    print("正在启动纯视觉小恐龙环境...")
    env = DinoVisionEnv()

    # 初始化环境，获取初始的连续 4 帧状态
    obs, info = env.reset()
    print(f"初始状态维度: {obs.shape}")  # 预期输出: (4, 84, 84)

    # 循环测试 500 步
    for step in range(500):
        # 随机采样动作：0(跑), 1(跳), 2(低头)
        action = env.action_space.sample()

        # 环境步进
        obs, reward, done, truncated, info = env.step(action)

        print(f"步数: {step:03d} | 动作: {action} | 奖励: {reward}")

        # ---------------------------------------------------------
        # 核心可视化部分：查看 Agent 眼中的世界
        # obs 维度是 (4, 84, 84)，我们提取最新的一帧（索引为 3 或 -1）
        latest_frame = obs[-1]

        # OpenCV 默认显示尺寸较小，我们将其放大 3 倍方便人类观察
        # interpolation=cv2.INTER_NEAREST 保持像素颗粒感，不模糊
        render_img = cv2.resize(latest_frame, (84 * 3, 84 * 3), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Agent Vision (84x84 Stacked)", render_img)
        # 等待 1 毫秒刷新图像界面
        cv2.waitKey(1)
        # ---------------------------------------------------------

        if done:
            print("--- 恐龙撞到了障碍物，重置环境 ---")
            obs, info = env.reset()
            # 死亡后暂停 1 秒，方便你看清撞击瞬间的画面
            time.sleep(1)

    # 测试结束，清理资源
    env.close()
    cv2.destroyAllWindows()
    print("视觉环境测试结束。")
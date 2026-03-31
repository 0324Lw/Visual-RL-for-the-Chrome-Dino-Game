import gymnasium as gym
from gymnasium import spaces
import numpy as np
from playwright.sync_api import sync_playwright
import os
import cv2
from collections import deque


class DinoVisionEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 动作空间：0=跑(无动作), 1=跳跃(空格键), 2=低头(向下方向键)
        self.action_space = spaces.Discrete(3)

        # 状态空间：连续 4 帧 84x84 的灰度图像
        # 采用 PyTorch 习惯的 Channel-First 格式 (Channels, Height, Width)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(4, 84, 84),
            dtype=np.uint8
        )

        # 使用双端队列来维护最近的 4 帧图像
        self.frames = deque(maxlen=4)

        # 启动 Playwright
        self.p = sync_playwright().start()
        self.browser = self.p.chromium.launch(headless=False)
        self.page = self.browser.new_page()

        # 设置固定的浏览器视口大小，确保每次截图比例一致
        self.page.set_viewport_size({"width": 800, "height": 600})

        current_dir = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(current_dir, 'index.html')
        game_url = f"file:///{html_path}".replace("\\", "/")

        self.page.goto(game_url)
        # 等待游戏画布加载完成
        self.canvas = self.page.locator(".runner-canvas")

    def _get_image(self):
        """核心视觉处理逻辑：截图 -> 灰度化 -> 裁剪 -> 缩放"""
        screenshot_bytes = self.canvas.screenshot()
        img_array = np.frombuffer(screenshot_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # ======== 【新增的这一行】 ========
        # 顺手把刚截出来的彩色原图保存下来，供测试脚本生成 GIF 使用，避免重复截图！
        self.latest_color_frame = img.copy()
        # ================================

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cropped = gray[0:150, 0:600]
        resized = cv2.resize(cropped, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 重置游戏
        self.page.evaluate("Runner.instance_.restart()")
        self.page.wait_for_timeout(100)  # 等待初始画面渲染

        # 获取初始帧
        init_frame = self._get_image()

        # 游戏刚开始时，把初始帧复制 4 次填满队列
        self.frames.clear()
        for _ in range(4):
            self.frames.append(init_frame)

        return np.stack(self.frames), {}

    def step(self, action):
        # 执行动作
        if action == 1:
            self.page.keyboard.press("Space")
        elif action == 2:
            self.page.keyboard.press("ArrowDown")

        # 步进时间：这是网络决策的频率。100ms 相当于 10 FPS
        self.page.wait_for_timeout(100)

        # 获取新一帧并加入队列 (最老的一帧会自动被挤出)
        new_frame = self._get_image()
        self.frames.append(new_frame)

        # 叠加状态
        obs = np.stack(self.frames)

        # 判定与奖励
        done = self.page.evaluate("Runner.instance_.crashed")
        reward = -10.0 if done else 0.1

        return obs, reward, done, False, {}

    def close(self):
        self.browser.close()
        self.p.stop()
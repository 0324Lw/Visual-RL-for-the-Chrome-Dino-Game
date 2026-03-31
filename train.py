import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from dino_env import DinoVisionEnv


# ==========================================
# 1. 优先经验回放 (PER) - SumTree 实现
# ==========================================
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write_idx = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def add(self, p, data):
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, p)
        self.write_idx = (self.write_idx + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get(self, s):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree): break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.capacity = capacity
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        priority = (self.max_priority) ** self.alpha
        self.tree.add(priority, data)

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.tree[0] / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.tree[0]
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones)), idxs, np.array(is_weight)

    def update_priorities(self, idxs, td_errors):
        for idx, td_error in zip(idxs, td_errors):
            p = (abs(td_error) + 1e-5) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.tree.n_entries


# ==========================================
# 2. 神经网络架构 (Dueling CNN + 正交初始化)
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
        # CNN 特征提取层
        self.conv = nn.Sequential(
            orthogonal_init(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), gain=np.sqrt(2)),
            nn.ReLU(),
            orthogonal_init(nn.Conv2d(32, 64, kernel_size=4, stride=2), gain=np.sqrt(2)),
            nn.ReLU(),
            orthogonal_init(nn.Conv2d(64, 64, kernel_size=3, stride=1), gain=np.sqrt(2)),
            nn.ReLU()
        )

        # 计算经过 CNN 后的特征维度 (对于 84x84 输入，通常是 64*7*7 = 3136)
        conv_out_size = self._get_conv_out(input_shape)

        # 优势流 (Advantage Stream)
        self.advantage = nn.Sequential(
            orthogonal_init(nn.Linear(conv_out_size, 512), gain=np.sqrt(2)),
            nn.ReLU(),
            orthogonal_init(nn.Linear(512, num_actions), gain=1.0)
        )

        # 价值流 (Value Stream)
        self.value = nn.Sequential(
            orthogonal_init(nn.Linear(conv_out_size, 512), gain=np.sqrt(2)),
            nn.ReLU(),
            orthogonal_init(nn.Linear(512, 1), gain=1.0)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # 将输入规范化到 [0, 1] 区间
        x = x.float() / 255.0
        features = self.conv(x)
        features = features.view(features.size(0), -1)

        adv = self.advantage(features)
        val = self.value(features)

        # Dueling 聚合公式: Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        q_vals = val + adv - adv.mean(dim=1, keepdim=True)
        return q_vals


# ==========================================
# 3. D3QN 智能体
# ==========================================
class D3QNAgent:
    def __init__(self, env, learning_rate=1e-4, gamma=0.99, batch_size=32, device='cuda'):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"使用计算设备: {self.device}")

        # 双网络架构
        self.online_net = DuelingCNN(env.observation_space.shape, env.action_space.n).to(self.device)
        self.target_net = DuelingCNN(env.observation_space.shape, env.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        # 学习率衰减 (每 10000 次 update 乘以 0.9)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.9)
        self.memory = PrioritizedReplayBuffer(capacity=20000)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return self.env.action_space.sample()

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state)
        return q_values.argmax().item()

    def update(self, beta):
        if len(self.memory) < self.batch_size:
            return None

        # 从 PER 中采样
        batch, idxs, is_weights = self.memory.sample(self.batch_size, beta)
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        is_weights = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        # 获取 Online 网络的 Q 值
        q_values = self.online_net(states).gather(1, actions)

        # Double DQN 核心逻辑
        with torch.no_grad():
            # 1. 用 Online 网络选出下一状态的最优动作 argmax Q_online(s', a)
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            # 2. 用 Target 网络计算该动作的 Q 值 Q_target(s', a*)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            # 3. 计算目标 Y 值
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # 计算 TD 误差用于更新 PER 优先级
        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy().flatten()
        self.memory.update_priorities(idxs, td_errors)

        # 结合 IS Weight 的加权 Huber Loss
        loss = (is_weights * F.smooth_l1_loss(q_values, target_q_values, reduction='none')).mean()

        # 梯度更新 + 梯度裁剪 (稳定机制)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 学习率步进
        self.lr_scheduler.step()

        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())


# ==========================================
# 4. 主训练循环
# ==========================================
def train(total_timesteps=100000):
    env = DinoVisionEnv()
    agent = D3QNAgent(env)

    # 探索率 epsilon 和 PER 的 beta 参数的退火机制
    beta_start, beta_frames = 0.4, total_timesteps
    epsilon_start = 1.0
    epsilon_end = 0.05

    # 让探索率在总步数的前 10% 内降到底 (比如 50万步就是前 5万步)
    exploration_steps = total_timesteps // 20

    def get_epsilon(step):
        # 线性衰减：每步减少固定的数值，直到触底
        decay_rate = (epsilon_start - epsilon_end) / exploration_steps
        epsilon = epsilon_start - step * decay_rate
        return max(epsilon_end, epsilon)  # 保证不低于 0.05

    def get_beta(step):
        return min(1.0, beta_start + step * (1.0 - beta_start) / beta_frames)

    save_dir = "./dino_models"
    os.makedirs(save_dir, exist_ok=True)

    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0

    history_rewards = []
    history_lengths = []

    print("开始训练...")
    start_time = time.time()

    for step in range(1, total_timesteps + 1):
        epsilon = get_epsilon(step)
        beta = get_beta(step)

        action = agent.select_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)

        agent.memory.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        episode_steps += 1

        # 每步都进行网络更新
        loss = agent.update(beta)

        if done:
            history_rewards.append(episode_reward)
            history_lengths.append(episode_steps)
            print(f"Step: {step}/{total_timesteps} | Episode: {len(history_rewards)} | "
                  f"Reward: {episode_reward:.1f} | Length: {episode_steps} | Epsilon: {epsilon:.3f}")
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0

        # 定期同步 Target 网络
        if step % 1000 == 0:
            agent.sync_target()

        # 定期保存模型
        if step % 20000 == 0:
            torch.save(agent.online_net.state_dict(), f"{save_dir}/d3qn_step_{step}.pth")
            print(f"[*] 模型已保存: d3qn_step_{step}.pth")

    env.close()
    print(f"训练结束！总耗时: {(time.time() - start_time) / 60:.2f} 分钟")

    # 绘制训练曲线
    plot_results(history_rewards, history_lengths)


def plot_results(rewards, lengths):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # 奖励曲线 (加一个简单的移动平均使其平滑)
    window = 10
    smoothed_rewards = [np.mean(rewards[max(0, i - window):(i + 1)]) for i in range(len(rewards))]

    axs[0].plot(rewards, alpha=0.3, color='blue', label='Raw Reward')
    axs[0].plot(smoothed_rewards, color='blue', label=f'MA ({window})')
    axs[0].set_title('Episode Rewards')
    axs[0].set_ylabel('Score')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(lengths, color='green')
    axs[1].set_title('Episode Lengths (Survival Time)')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Steps')
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.show()


if __name__ == '__main__':
    # 设定总训练步数 (对于视觉任务，10万步只是热身，若想完全收敛通常需要 50万步以上)
    train(total_timesteps=500000)
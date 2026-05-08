import torch
import numpy as np
import matplotlib.pyplot as plt


class SigmoidWeightScheduler:
    def __init__(self,
                 min_weight=0.0,
                 max_weight=0.1,
                 warmup_steps=0,
                 max_steps=5000,
                 center_step=2000,
                 sharpness=10.0):
        """
        Sigmoid 形状的权重调度器

        Args:
            min_weight (float): 初始权重 (通常为0或很小)
            max_weight (float): 最终权重 (例如 0.1)
            warmup_steps (int): 在此步数之前，权重强制为 min_weight
            max_steps (int): 总训练步数 (用于归一化)
            center_step (int): Sigmoid 曲线的中心点 (权重达到 (min+max)/2 的步数)
            sharpness (float): 控制 Sigmoid 的陡峭程度 (曲率)。
                               值越大越陡峭，值越小越接近线性。
        """
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.warmup_steps = warmup_steps
        self.center_step = center_step

        # 我们对输入进行缩放，使得 sharpness 参数更直观
        # 输入 x 从 [0, max_steps] 映射
        self.scale_factor = sharpness / (max_steps / 2)

    def get_weight(self, current_step):
        # 1. Warmup 阶段
        if current_step < self.warmup_steps:
            return self.min_weight

        # 2. 计算 Sigmoid 输入
        # x = 0 时 (即 current_step = center_step)，sigmoid 输出 0.5
        x = (current_step - self.center_step) * self.scale_factor

        # 3. 计算 Sigmoid (使用 numpy 或 torch 均可)
        # sigmoid(x) = 1 / (1 + exp(-x))
        # 考虑到数值稳定性，处理一下大数值
        if x > 20:
            sig = 1.0
        elif x < -20:
            sig = 0.0
        else:
            sig = 1.0 / (1.0 + np.exp(-x))

        # 4. 映射到 [min, max]
        weight = self.min_weight + (self.max_weight - self.min_weight) * sig

        return weight

    def plot_schedule(self, max_steps=5000):
        """辅助函数：画出曲线看看形状对不对"""
        steps = np.arange(max_steps)
        weights = [self.get_weight(s) for s in steps]

        plt.figure(figsize=(8, 4))
        plt.plot(steps, weights, label='Eikonal Weight')
        plt.axvline(x=self.center_step, color='r', linestyle='--', label='Center')
        plt.title('Eikonal Loss Weight Schedule')
        plt.xlabel('Step')
        plt.ylabel('Weight')
        plt.grid(True)
        plt.legend()
        plt.show()


# === 测试代码 ===
if __name__ == "__main__":
    # 配置你的场景
    max_steps=1000
    scheduler = SigmoidWeightScheduler(
        min_weight=0.0000,  # 开始几乎不约束，防止梯度爆炸
        max_weight=0.01,  # 最终目标权重
        warmup_steps=0,  # 前500步让网络自由奔跑，先学UDF形状
        max_steps=max_steps,  # 总步数
        center_step=max_steps/2,  # 在第2000步时增长到一半 (0.05)
        sharpness=10  # 陡峭程度
    )

    # 画图看看
    scheduler.plot_schedule(max_steps)
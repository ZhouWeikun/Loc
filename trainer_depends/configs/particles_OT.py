import torch


def sinkhorn_algorithm(cost_matrix, epsilon=0.1, n_iters=50):
    # 简化的 Sinkhorn 迭代，参考论文 Algorithm 2 [cite: 206]
    N = cost_matrix.shape[0]

    # 对数域计算以保持数值稳定
    K = torch.exp(-cost_matrix / epsilon)

    u = torch.ones(N, device=cost_matrix.device) / N
    v = torch.ones(N, device=cost_matrix.device) / N

    # 目标权重 (你的特征相似度 + UDF)
    # 假设 weights 已经归一化

    for _ in range(n_iters):
        # 迭代更新势能向量 u, v
        # 注意：这里需要严格按照 OT 的对偶问题求解，或者使用 geomloss 库
        # 这里仅作逻辑演示，核心是让行和接近 1/N，列和接近 weights
        pass

        # 计算传输矩阵 P
    # P_ij 意味着从 i 移动到 j 的“质量”
    return P


def dpf_update(particles, udf_field, feature_field, target_feature):
    """
    particles: [N, 4] (x, y, s, d)
    """
    N = particles.shape[0]

    # 1. 计算权重 (Gravity Magnitude)
    # 从 feature field 采样特征
    sampled_feats = feature_field.sample(particles)  # [N, D]

    # 计算相似度 (Feature Gravity)
    sim_scores = torch.exp(-torch.norm(sampled_feats - target_feature, dim=1) ** 2)

    # 计算 UDF 约束 (Geometric Gravity)
    udf_vals = udf_field.sample(particles)
    geo_scores = torch.exp(-torch.abs(udf_vals))

    # 综合权重
    weights = sim_scores * geo_scores
    weights = weights / weights.sum()  # 归一化

    # 2. 计算代价矩阵 (Cost Matrix)
    # 粒子间的距离矩阵
    dist_matrix = torch.cdist(particles, particles, p=2) ** 2  # [N, N]

    # 3. 计算传输矩阵 (Transport Matrix)
    # 使用 Sinkhorn 求解最优传输方案
    # 这里可以使用 GeomLoss 库来实现高效的 Sinkhorn
    # P 形状为 [N, N]
    import geomloss
    loss = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=0.05, potentials=True)
    # GeomLoss通常返回势能，我们需要传输矩阵，或者直接利用它的梯度性质
    # 也可以手动实现简单的 Sinkhorn 得到 P

    # 假设得到了 P (Transport Matrix)
    # P[i, j] 表示粒子 i 有多少成分应该变成粒子 j 的位置

    # 4. 位移 (Displacement)
    # new_particles[i] = N * sum_k (P[i, k] * old_particles[k])
    # 这实际上是一个矩阵乘法
    # 注意：论文公式是 \tilde{X} = N * P * X

    new_particles = N * torch.matmul(P, particles)

    return new_particles
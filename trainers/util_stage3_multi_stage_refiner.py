from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as TF


@dataclass
class SeedRegionConfig:
    """
    Stage 1 配置：从 coarse seeds 出发，做初筛、局部邻域采样、迭代重定位与 seed-cloud 淘汰。
    """

    n_bins_4d: Sequence[int]
    topN_seed: int = 64
    # 基础局部半径系数，r_d = alpha * Delta_d / 2
    alpha: float = 1.0
    # Stage 1 每轮每个 seed-cloud 采样多少个点，可按轮变化
    samples_per_round: Sequence[int] = (32,)
    # Stage 1 每轮采样范围系数，相对基础半径缩放，可按轮变化
    radius_scale_per_round: Sequence[float] = (1.0,)
    # 每轮用 top-q 比例 elite 点做聚合
    elite_ratio: float = 0.25
    # 每轮保留多少比例 seed-cloud，可按轮变化
    survival_ratio_per_round: Sequence[float] = (1.0,)
    # 每轮至少保留多少个 seed-cloud
    min_surviving_clouds: int = 1
    # 邻域采样方式，当前支持 uniform / sobol / sobol_deterministic
    local_sample_method: str = "sobol"
    # relocation 权重模式，如 softmax / linear
    relocate_weight_mode: str = "softmax"
    relocate_weight_beta: float = 10.0
    # True 时 Stage 1 重采样包含 scale 维；False 时固定为当前 center 的 scale
    enable_scale_sampling: bool = True


@dataclass
class Stage3CMAConfig:
    """
    Stage 3 配置：只让少量 surviving modes 进入最终 CMA-ES 精修。

    当前约定：
    - CMA 初始中心直接使用当前 mode center
    - CMA 初始 sigma 优先使用手动值，否则使用当前 mode 的 sigma_diag
    """

    enable: bool = True
    # 最多选多少个 mode 进入 CMA
    cma_max_input_mode: int = 64
    # Stage 3 最终精修时使用的打分模式，如 ingp / projector / product
    score_mode: str = "ingp"
    # 是否让 CMA 同时优化 scale 维；False 时只优化 [nr, nc, rot]
    cma_optimize_scale: bool = True
    # CMA 变体名称，常见为标准 CMA 或对角近似变体
    cma_variant: str = "CMA"
    # Stage 3 初始 sigma 的来源，当前默认直接使用当前 mode 的对角尺度
    init_sigma_source: str = "mode_diag"
    # 若当前 mode 给的是逐维尺度，如何压缩成 CMA 所需的初始标量 sigma
    init_sigma_reduce: str = "mean"
    # 手动指定 Stage 3 初始 sigma；若不为 None，则优先级最高
    init_sigma_manual: Optional[float] = None
    # 对当前 mode 推导出的初始 sigma 再乘一个缩放因子
    init_sigma_scale: float = 1.0
    # 安全兜底：当当前 mode 尺度不可用或异常时，默认使用 Stage 1
    # 前两个维度采样间隔 Delta_x / Delta_y 的均值；若显式给定数值，则覆盖该默认行为
    init_sigma_fallback: Optional[float] = None
    # 每一代采样的候选个体数
    popsize: int = 64
    # CMA 总迭代代数
    n_iterations: int = 8
    # 是否在多代无提升时提前停止
    enable_early_stop: bool = False
    # 早停耐心值：连续多少代无明显提升就停止
    early_stop_patience: int = 3
    # 是否启用 Stage 3 mode 间竞争淘汰
    enable_competition: bool = False
    # 在 warmup 之后，每隔多少代做一次比较和淘汰
    competition_interval: int = 2
    # 每轮保留多少比例 mode
    survival_ratio: float = 0.5
    # 每轮至少保留多少个 mode
    min_surviving_modes: int = 1
    # 当前采样点中，取 top-q 比例作为 elite
    elite_ratio: float = 0.25
    # elite 样本做加权聚合时的 softmax beta
    metric_weight_beta: float = 10.0
    # True 时，在 Stage 3 结束后，对每个 mode 的两个候选点做 winner 比较：
    # {Stage3-CMA best, Stage1.5 best}。
    # 若 Stage1 / Stage3 使用同一评分模式，则直接复用已保存分数；
    # 否则用 Stage3 评分函数对两者重新打分。
    rerank_per_mode_after_stage3: bool = False

    @property
    def max_modes_to_cma(self) -> int:
        return int(self.cma_max_input_mode)


@dataclass
class Stage4GradConfig:
    """
    Stage 4 配置：对上一阶段 top-K 坐标做基于梯度的局部优化。
    """

    enable: bool = False
    # 输入来源，默认取最新有效阶段
    input_stage: str = "latest"
    # 最多取多少个 top-K 坐标送入 Stage 4
    topk_input: int = 32
    # Stage 4 最终输出分数使用的评估模式
    score_mode: str = "ingp"
    # Stage 4 梯度优化使用的坐标空间：'raw' 或 'linear'
    opt_space: str = "raw"
    # 梯度优化步数
    n_steps: int = 100
    # 学习率
    lr_xy: float = 1e-5
    lr_rot: float = 5e-6
    lr_scale: float = 1e-5
    # 是否优化 scale 维
    optimize_scale: bool = True
    # 梯度裁剪上限
    max_grad_norm: float = 1.0
    # 是否打印 Stage 4 内部优化日志
    verbose: bool = False
    # Stage 4 内部优化日志打印间隔
    log_interval: int = 10


@dataclass
class SeedModeSearchConfig:
    """
    多阶段搜索总配置。

    负责串起：
    Stage 1 seed screening / seed-cloud
    Stage 1.5 relocation
    Stage 3 final CMA refinement
    Stage 4 gradient refinement
    """

    score_mode_stage1: str = "ingp"
    score_mode_stage3: str = "ingp"
    metric_goal: str = "maximize"
    seed_region: SeedRegionConfig = field(default_factory=lambda: SeedRegionConfig(n_bins_4d=(40, 30, 36, 1)))
    stage3: Stage3CMAConfig = field(default_factory=Stage3CMAConfig)
    stage4: Stage4GradConfig = field(default_factory=Stage4GradConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SeedCandidate:
    """
    Coarse seed after initial screening.
    """

    query_id: int
    seed_id: str
    coord_raw: torch.Tensor
    seed_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SeedCloud:
    """
    Local region samples around one seed.
    """

    query_id: int
    seed_id: str
    center_raw: torch.Tensor
    radius_raw: torch.Tensor
    sample_coords_raw: torch.Tensor
    sample_scores: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModeState:
    """
    Candidate local mode produced from one or more seed-cloud summaries.
    """

    query_id: int
    mode_id: str
    center_raw: torch.Tensor
    sigma_diag_raw: torch.Tensor
    score_mode: str
    stage: str = "stage1.5"
    alive: bool = True
    budget_used: float = 0.0
    best_coord_raw: Optional[torch.Tensor] = None
    best_score: Optional[float] = None
    latest_metric: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def clone(self) -> "ModeState":
        return ModeState(
            query_id=self.query_id,
            mode_id=self.mode_id,
            center_raw=self.center_raw.clone(),
            sigma_diag_raw=self.sigma_diag_raw.clone(),
            score_mode=self.score_mode,
            stage=self.stage,
            alive=bool(self.alive),
            budget_used=float(self.budget_used),
            best_coord_raw=None if self.best_coord_raw is None else self.best_coord_raw.clone(),
            best_score=None if self.best_score is None else float(self.best_score),
            latest_metric=None if self.latest_metric is None else float(self.latest_metric),
            metadata=dict(self.metadata),
            history=list(self.history),
        )

    def record_event(self, event_name: str, **kwargs: Any) -> None:
        event = {"event": event_name, "stage": self.stage, "budget_used": float(self.budget_used)}
        event.update(kwargs)
        self.history.append(event)


@dataclass
class StageTopKRecord:
    """
    One stage's ranked top-K outputs for a single query.
    """

    stage_id: int
    stage_name: str
    score_func_name: str
    coords_topk_raw: torch.Tensor
    scores_topk: torch.Tensor
    mode_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryStageTrace:
    """
    Per-query stage trace used to hand top-K results to later optional stages.
    """

    query_id: int
    stage_records: List[StageTopKRecord] = field(default_factory=list)

    def add_record(self, record: StageTopKRecord) -> None:
        self.stage_records.append(record)

    def latest_record(self) -> Optional[StageTopKRecord]:
        valid_records = [record for record in self.stage_records if int(record.coords_topk_raw.shape[0]) > 0]
        if len(valid_records) == 0:
            return None
        return max(valid_records, key=lambda record: int(record.stage_id))


@dataclass
class SeedModeSearchResult:
    """
    Full result for one query after Stage 1 and Stage 3.
    """

    query_id: int
    seeds_kept: List[SeedCandidate]
    seed_clouds: List[SeedCloud]
    modes_init: List[ModeState]
    modes_before_stage3: List[ModeState]
    modes_final: List[ModeState]
    stage_trace: QueryStageTrace
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSeedScreeningStrategy(ABC):
    @abstractmethod
    def select_seeds(
        self,
        query_id: int,
        coarse_seed_coords_raw: torch.Tensor,
        coarse_seed_scores: torch.Tensor,
        config: SeedModeSearchConfig,
        query_context: Optional[Dict[str, Any]] = None,
    ) -> List[SeedCandidate]:
        raise NotImplementedError


class BaseSeedCloudBuilder(ABC):
    @abstractmethod
    def build_seed_clouds(
        self,
        seeds: List[SeedCandidate],
        config: SeedModeSearchConfig,
        query_context: Optional[Dict[str, Any]] = None,
    ) -> List[SeedCloud]:
        raise NotImplementedError


class BaseSeedCloudRelocator(ABC):
    @abstractmethod
    def relocate_seed_clouds(
        self,
        seed_clouds: List[SeedCloud],
        config: SeedModeSearchConfig,
        query_context: Optional[Dict[str, Any]] = None,
    ) -> List[ModeState]:
        raise NotImplementedError


class BaseModeDeduper(ABC):
    @abstractmethod
    def dedup_modes(
        self,
        modes_init: List[ModeState],
        config: SeedModeSearchConfig,
    ) -> List[ModeState]:
        raise NotImplementedError


class BaseFinalModeOptimizer(ABC):
    @abstractmethod
    def optimize_modes(
        self,
        query_id: int,
        modes: List[ModeState],
        config: SeedModeSearchConfig,
        query_context: Optional[Dict[str, Any]] = None,
    ) -> List[ModeState]:
        raise NotImplementedError


class BaseStage4Optimizer(ABC):
    @abstractmethod
    def optimize_from_trace(
        self,
        query_id: int,
        stage_trace: QueryStageTrace,
        config: SeedModeSearchConfig,
        query_context: Optional[Dict[str, Any]] = None,
    ) -> List[ModeState]:
        raise NotImplementedError


def _as_float_tensor(x: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.float32) if device is not None else x.to(dtype=torch.float32)
    return torch.tensor(x, device=device, dtype=torch.float32)


def _wrap_theta_raw(theta_raw: torch.Tensor) -> torch.Tensor:
    return torch.remainder(theta_raw + torch.pi, 2 * torch.pi) - torch.pi


def _resolve_score_fn(query_context: Optional[Dict[str, Any]], key: str = "score_coords_fn"):
    if query_context is None or key not in query_context:
        raise KeyError(f"query_context must provide a callable `{key}(coords_raw_nx4)`.")
    score_fn = query_context[key]
    if not callable(score_fn):
        raise TypeError(f"query_context['{key}'] must be callable.")
    return score_fn


def _require_query_context_value(query_context: Optional[Dict[str, Any]], key: str):
    if query_context is None or key not in query_context or query_context[key] is None:
        raise KeyError(f"query_context must provide `{key}`.")
    return query_context[key]


def _resolve_stage1_bin_sizes_raw(
    config: SeedModeSearchConfig,
    query_context: Optional[Dict[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    if query_context is not None:
        for key in ("stage1_bin_sizes_raw", "coarse_bin_sizes_raw"):
            if key in query_context and query_context[key] is not None:
                sizes = _as_float_tensor(query_context[key], device=device).reshape(-1)
                if sizes.numel() != 4:
                    raise ValueError(f"{key} must provide 4 values, got shape {tuple(sizes.shape)}")
                return sizes

    if query_context is None or "coords_processor" not in query_context:
        raise KeyError(
            "query_context must provide either `stage1_bin_sizes_raw` / `coarse_bin_sizes_raw` "
            "or `coords_processor`."
        )

    processor = query_context["coords_processor"]
    n_nr, n_nc, n_rot, n_scale = [max(1, int(v)) for v in config.seed_region.n_bins_4d]
    delta_nr = float(processor.nrc_diff[0].item()) / n_nr
    delta_nc = float(processor.nrc_diff[1].item()) / n_nc
    delta_rot = float(2.0 * math.pi) / n_rot
    scale_min = float(torch.exp(processor.scale_log_min).item())
    scale_max = float(torch.exp(processor.scale_log_max).item())
    delta_scale = (scale_max - scale_min) / max(1, n_scale)
    return torch.tensor([delta_nr, delta_nc, delta_rot, delta_scale], device=device, dtype=torch.float32)


def _resolve_raw_bounds(
    query_context: Optional[Dict[str, Any]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if query_context is not None and "raw_bounds_low" in query_context and "raw_bounds_high" in query_context:
        low = _as_float_tensor(query_context["raw_bounds_low"], device=device).reshape(4)
        high = _as_float_tensor(query_context["raw_bounds_high"], device=device).reshape(4)
        return torch.stack([low, high], dim=0)

    if query_context is not None and "coords_processor" in query_context:
        processor = query_context["coords_processor"]
        low = torch.tensor(
            [
                float(processor.nrc_min[0].item()),
                float(processor.nrc_min[1].item()),
                -math.pi,
                float(torch.exp(processor.scale_log_min).item()),
            ],
            device=device,
            dtype=torch.float32,
        )
        high = torch.tensor(
            [
                float(processor.nrc_max[0].item()),
                float(processor.nrc_max[1].item()),
                math.pi,
                float(torch.exp(processor.scale_log_max).item()),
            ],
            device=device,
            dtype=torch.float32,
        )
        return torch.stack([low, high], dim=0)

    return None


def _project_raw_coords(
    coords_raw: torch.Tensor,
    query_context: Optional[Dict[str, Any]],
) -> torch.Tensor:
    coords = coords_raw
    device = coords.device
    bounds = _resolve_raw_bounds(query_context, device=device)
    x = coords[..., 0:1]
    y = coords[..., 1:2]
    theta = _wrap_theta_raw(coords[..., 2:3])
    scale = coords[..., 3:4]
    if bounds is None:
        scale = scale.clamp(min=1e-6)
        return torch.cat([x, y, theta, scale], dim=-1)
    x = x.clamp(bounds[0, 0], bounds[1, 0])
    y = y.clamp(bounds[0, 1], bounds[1, 1])
    scale = scale.clamp(bounds[0, 3], bounds[1, 3])
    return torch.cat([x, y, theta, scale], dim=-1)


def _project_linear_coords(coords_linear: torch.Tensor) -> torch.Tensor:
    if coords_linear.shape[-1] != 4:
        raise ValueError(f"coords_linear must end with dim 4, got {tuple(coords_linear.shape)}")
    x = torch.clamp(coords_linear[..., 0:1], min=-1.0, max=1.0)
    y = torch.clamp(coords_linear[..., 1:2], min=-1.0, max=1.0)
    theta = torch.remainder(coords_linear[..., 2:3] + 1.0, 2.0) - 1.0
    scale = torch.clamp(coords_linear[..., 3:4], min=-1.0, max=1.0)
    return torch.cat([x, y, theta, scale], dim=-1)


def _sample_box_offsets(
    n_points: int,
    radius_raw: torch.Tensor,
    method: str,
    device: torch.device,
) -> torch.Tensor:
    n_points = int(n_points)
    if n_points <= 0:
        return torch.zeros((0, 4), device=device, dtype=torch.float32)

    method = str(method).strip().lower()
    if method == "sobol":
        engine = torch.quasirandom.SobolEngine(dimension=4, scramble=True)
        unit = engine.draw(n_points).to(device=device, dtype=torch.float32)
    elif method in ("sobol_deterministic", "sobol_det", "deterministic_sobol"):
        engine = torch.quasirandom.SobolEngine(dimension=4, scramble=False)
        unit = engine.draw(n_points).to(device=device, dtype=torch.float32)
    elif method in ("uniform", "random"):
        unit = torch.rand((n_points, 4), device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported sampling method: {method}")
    return (unit * 2.0 - 1.0) * radius_raw.reshape(1, 4)


def _score_raw_coords(
    coords_raw: torch.Tensor,
    query_context: Optional[Dict[str, Any]],
    score_key: str = "score_coords_fn",
) -> torch.Tensor:
    score_fn = _resolve_score_fn(query_context, key=score_key)
    scores = score_fn(coords_raw)
    scores_t = _as_float_tensor(scores, device=coords_raw.device).reshape(-1)
    return torch.nan_to_num(scores_t, nan=-1e9, posinf=1e9, neginf=-1e9)


def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    w = weights.reshape(-1).clamp(min=0.0)
    denom = w.sum()
    if float(denom.item()) <= 0:
        return torch.ones_like(w) / max(1, w.numel())
    return w / denom


def _compute_weighted_center_raw(coords_raw: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    coords = coords_raw.reshape(-1, 4)
    w = _normalize_weights(weights).reshape(-1, 1)
    center_xy = torch.sum(w * coords[:, 0:2], dim=0)
    theta = coords[:, 2]
    theta_cos = torch.sum(w[:, 0] * torch.cos(theta))
    theta_sin = torch.sum(w[:, 0] * torch.sin(theta))
    center_theta = torch.atan2(theta_sin, theta_cos).reshape(1)
    scale = coords[:, 3].clamp(min=1e-6)
    center_scale = torch.exp(torch.sum(w[:, 0] * torch.log(scale))).reshape(1)
    return torch.cat([center_xy, center_theta, center_scale], dim=0)


def _normalize_weights_batched(weights: torch.Tensor) -> torch.Tensor:
    w = weights.clamp(min=0.0)
    denom = w.sum(dim=-1, keepdim=True)
    fallback = torch.ones_like(w) / max(1, int(w.shape[-1]))
    return torch.where(denom > 0, w / denom.clamp(min=1e-12), fallback)


def _compute_weighted_center_raw_batched(coords_raw: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    coords = _as_float_tensor(coords_raw)
    if coords.ndim != 3 or coords.shape[-1] != 4:
        raise ValueError("coords_raw must be [B, K, 4].")
    w = _normalize_weights_batched(_as_float_tensor(weights, device=coords.device)).reshape(coords.shape[0], -1, 1)
    if int(w.shape[1]) != int(coords.shape[1]):
        raise ValueError("weights must have shape [B, K] matching coords_raw.")

    center_xy = torch.sum(w * coords[:, :, 0:2], dim=1)
    theta = coords[:, :, 2]
    theta_cos = torch.sum(w[:, :, 0] * torch.cos(theta), dim=1)
    theta_sin = torch.sum(w[:, :, 0] * torch.sin(theta), dim=1)
    center_theta = torch.atan2(theta_sin, theta_cos).reshape(-1, 1)
    scale = coords[:, :, 3].clamp(min=1e-6)
    center_scale = torch.exp(torch.sum(w[:, :, 0] * torch.log(scale), dim=1)).reshape(-1, 1)
    return torch.cat([center_xy, center_theta, center_scale], dim=-1)


def _compute_sigma_diag_raw(
    coords_raw: torch.Tensor,
    center_raw: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    coords = coords_raw.reshape(-1, 4)
    center = center_raw.reshape(1, 4)
    if weights is None:
        w = torch.ones((coords.shape[0],), device=coords.device, dtype=torch.float32)
    else:
        w = _normalize_weights(weights)
    diff_xy = coords[:, 0:2] - center[:, 0:2]
    diff_theta = _wrap_theta_raw(coords[:, 2] - center[:, 2]).reshape(-1, 1)
    diff_scale = (coords[:, 3:4] - center[:, 3:4])
    diff = torch.cat([diff_xy, diff_theta, diff_scale], dim=-1)
    var = torch.sum(w.reshape(-1, 1) * diff.pow(2.0), dim=0)
    return torch.sqrt(var.clamp(min=1e-12))


def _compute_weighted_elite_metric(
    scores: torch.Tensor,
    elite_ratio: float,
    metric_goal: str,
    beta: float,
) -> float:
    scores_t = _as_float_tensor(scores).reshape(-1)
    if scores_t.numel() == 0:
        return -1e9 if metric_goal == "maximize" else 1e9

    elite_count = max(1, int(math.ceil(int(scores_t.numel()) * float(elite_ratio))))
    reverse = metric_goal == "maximize"
    elite_idx = torch.argsort(scores_t, descending=reverse)[:elite_count]
    elite_scores = scores_t.index_select(0, elite_idx)

    if reverse:
        logits = elite_scores - elite_scores.max()
    else:
        logits = -(elite_scores - elite_scores.min())
    weights = torch.softmax(float(beta) * logits, dim=0)
    return float(torch.sum(weights * elite_scores).item())


def _resolve_stage1_round_count(config: SeedModeSearchConfig) -> int:
    return max(
        1,
        len(list(config.seed_region.samples_per_round)),
        len(list(config.seed_region.radius_scale_per_round)),
        len(list(config.seed_region.survival_ratio_per_round)),
    )


def _resolve_stage1_sample_count(config: SeedModeSearchConfig, round_index: int) -> int:
    schedule = list(config.seed_region.samples_per_round)
    if len(schedule) == 0:
        raise ValueError("seed_region.samples_per_round must contain at least one positive integer.")
    idx = min(int(round_index), len(schedule) - 1)
    n_samples = int(schedule[idx])
    if n_samples <= 0:
        raise ValueError("seed_region.samples_per_round values must be > 0.")
    return n_samples


def _resolve_stage1_radius_scale(config: SeedModeSearchConfig, round_index: int) -> float:
    schedule = list(config.seed_region.radius_scale_per_round)
    if len(schedule) == 0:
        raise ValueError("seed_region.radius_scale_per_round must contain at least one positive value.")
    idx = min(int(round_index), len(schedule) - 1)
    radius_scale = float(schedule[idx])
    if radius_scale <= 0:
        raise ValueError("seed_region.radius_scale_per_round values must be > 0.")
    return radius_scale


def _resolve_stage1_survival_ratio(config: SeedModeSearchConfig, round_index: int) -> float:
    schedule = list(config.seed_region.survival_ratio_per_round)
    if len(schedule) == 0:
        raise ValueError("seed_region.survival_ratio_per_round must contain at least one value.")
    idx = min(int(round_index), len(schedule) - 1)
    survival_ratio = float(schedule[idx])
    if survival_ratio <= 0 or survival_ratio > 1.0:
        raise ValueError("seed_region.survival_ratio_per_round values must be in (0, 1].")
    return survival_ratio


def _resolve_stage1_radius_raw(
    config: SeedModeSearchConfig,
    query_context: Optional[Dict[str, Any]],
    device: torch.device,
    round_index: int,
) -> torch.Tensor:
    delta_raw = _resolve_stage1_bin_sizes_raw(config=config, query_context=query_context, device=device)
    radius_raw = float(config.seed_region.alpha) * _resolve_stage1_radius_scale(config, round_index) * delta_raw / 2.0
    if not bool(config.seed_region.enable_scale_sampling):
        radius_raw = radius_raw.clone()
        radius_raw[3] = 0.0
    return radius_raw


def _score_key_stage1(query_context: Optional[Dict[str, Any]]) -> str:
    if query_context is not None and "score_coords_fn_stage1" in query_context:
        return "score_coords_fn_stage1"
    return "score_coords_fn"


def _resolve_score_chunk_size(
    query_context: Optional[Dict[str, Any]],
    key: str,
) -> Optional[int]:
    if query_context is None or key not in query_context or query_context[key] is None:
        return None
    chunk_size = int(query_context[key])
    return chunk_size if chunk_size > 0 else None


def _score_raw_coords_chunked(
    coords_raw: torch.Tensor,
    query_context: Optional[Dict[str, Any]],
    score_key: str,
    chunk_key: str,
) -> torch.Tensor:
    chunk_size = _resolve_score_chunk_size(query_context, key=chunk_key)
    if chunk_size is None or int(coords_raw.shape[0]) <= chunk_size:
        return _score_raw_coords(coords_raw, query_context=query_context, score_key=score_key)

    scores_parts: List[torch.Tensor] = []
    for start in range(0, int(coords_raw.shape[0]), chunk_size):
        end = min(start + chunk_size, int(coords_raw.shape[0]))
        scores_parts.append(
            _score_raw_coords(
                coords_raw[start:end],
                query_context=query_context,
                score_key=score_key,
            )
        )
    return torch.cat(scores_parts, dim=0)


def _summarize_stage4_scores(
    coords_raw: torch.Tensor,
    config: SeedModeSearchConfig,
    query_context: Optional[Dict[str, Any]],
) -> Tuple[float, float]:
    score_key = "score_coords_fn_stage4" if query_context is not None and "score_coords_fn_stage4" in query_context else "score_coords_fn"
    scores_t = _score_raw_coords_chunked(
        coords_raw.reshape(-1, 4),
        query_context=query_context,
        score_key=score_key,
        chunk_key="stage4_score_chunk_size",
    ).reshape(-1)
    score_mean = float(scores_t.mean().item()) if scores_t.numel() > 0 else float("nan")
    if config.metric_goal == "maximize":
        score_top1 = float(scores_t.max().item()) if scores_t.numel() > 0 else float("nan")
    else:
        score_top1 = float(scores_t.min().item()) if scores_t.numel() > 0 else float("nan")
    return score_mean, score_top1


class BatchedBoxSampler:
    """
    GPU batched local box sampler for Stage 1 seed-cloud construction.
    """

    @staticmethod
    def sample_around_centers(
        centers_raw: torch.Tensor,
        radius_raw: torch.Tensor,
        n_samples: int,
        method: str,
        query_context: Optional[Dict[str, Any]] = None,
        enable_scale_sampling: bool = True,
    ) -> torch.Tensor:
        centers = _as_float_tensor(centers_raw)
        if centers.ndim != 2 or centers.shape[-1] != 4:
            raise ValueError("centers_raw must be [N, 4].")

        device = centers.device
        n_regions = int(centers.shape[0])
        n_samples = max(1, int(n_samples))
        n_rand = max(0, n_samples - 1)

        radius = _as_float_tensor(radius_raw, device=device)
        if radius.ndim == 1:
            radius = radius.reshape(1, 4).expand(n_regions, 4).clone()
        elif radius.ndim == 2 and tuple(radius.shape) == (n_regions, 4):
            radius = radius.clone()
        else:
            raise ValueError("radius_raw must be [4] or [N, 4].")

        if not enable_scale_sampling:
            radius[:, 3] = 0.0

        coords = centers[:, None, :].expand(n_regions, n_samples, 4).clone()
        if n_rand > 0:
            flat_radius = radius[:, None, :].expand(n_regions, n_rand, 4).reshape(-1, 4)
            flat_offsets = _sample_box_offsets(
                n_points=int(flat_radius.shape[0]),
                radius_raw=torch.ones((4,), device=device, dtype=torch.float32),
                method=method,
                device=device,
            ) * flat_radius
            offsets = flat_offsets.reshape(n_regions, n_rand, 4)
            coords[:, 1:, :] = centers[:, None, :] + offsets

        coords = _project_raw_coords(coords.reshape(-1, 4), query_context=query_context).reshape(n_regions, n_samples, 4)
        if not enable_scale_sampling:
            coords[:, :, 3] = centers[:, None, 3]
        return coords


def _resolve_stage3_init_sigma(
    mode_state: ModeState,
    config: SeedModeSearchConfig,
    query_context: Optional[Dict[str, Any]],
) -> float:
    if config.stage3.init_sigma_manual is not None:
        sigma_manual = float(config.stage3.init_sigma_manual)
        return max(1e-4, sigma_manual)

    source = str(config.stage3.init_sigma_source).strip().lower()
    reduce_mode = str(config.stage3.init_sigma_reduce).strip().lower()
    sigma_diag = None
    if source in ("mode_diag", "stage2_diag"):
        sigma_diag = mode_state.sigma_diag_raw
    else:
        raise ValueError(f"Unsupported init_sigma_source: {config.stage3.init_sigma_source}")

    if sigma_diag is None or sigma_diag.numel() == 0:
        sigma_scalar = None
    else:
        sigma_diag = _as_float_tensor(sigma_diag, device=mode_state.center_raw.device).reshape(-1)
        if reduce_mode == "mean":
            sigma_scalar = float(sigma_diag.mean().item())
        elif reduce_mode == "max":
            sigma_scalar = float(sigma_diag.max().item())
        elif reduce_mode == "xy_mean":
            sigma_scalar = float(sigma_diag[:2].mean().item())
        else:
            raise ValueError(f"Unsupported init_sigma_reduce: {config.stage3.init_sigma_reduce}")

    if sigma_scalar is None or not math.isfinite(sigma_scalar) or sigma_scalar <= 0:
        if config.stage3.init_sigma_fallback is not None:
            sigma_scalar = float(config.stage3.init_sigma_fallback)
        else:
            stage1_delta = _resolve_stage1_bin_sizes_raw(
                config=config,
                query_context=query_context,
                device=mode_state.center_raw.device,
            )
            sigma_scalar = float(stage1_delta[:2].mean().item())

    sigma_scalar = float(config.stage3.init_sigma_scale) * sigma_scalar
    return max(1e-4, sigma_scalar)


class TopNSeedScreening(BaseSeedScreeningStrategy):
    """
    Keep the highest-scoring coarse seeds before local region construction.
    """

    def select_seeds(
        self,
        query_id: int,
        coarse_seed_coords_raw: torch.Tensor,
        coarse_seed_scores: torch.Tensor,
        config: SeedModeSearchConfig,
        query_context: Optional[Dict[str, Any]] = None,
    ) -> List[SeedCandidate]:
        coords = _as_float_tensor(coarse_seed_coords_raw).reshape(-1, 4)
        scores = _as_float_tensor(coarse_seed_scores, device=coords.device).reshape(-1)
        if coords.shape[0] != scores.shape[0]:
            raise ValueError("coarse_seed_coords_raw and coarse_seed_scores must have the same length.")
        if coords.shape[0] == 0:
            return []

        topk = min(int(config.seed_region.topN_seed), int(coords.shape[0]))
        rank_idx = torch.argsort(scores, descending=(config.metric_goal == "maximize"))[:topk]
        seeds = []
        for rank, idx in enumerate(rank_idx.tolist()):
            seeds.append(
                SeedCandidate(
                    query_id=query_id,
                    seed_id=f"q{query_id}_seed{rank}",
                    coord_raw=coords[idx].clone(),
                    seed_score=float(scores[idx].item()),
                    metadata={"rank": rank, "source_index": int(idx)},
                )
            )
        return seeds


class LocalSeedCloudBuilder(BaseSeedCloudBuilder):
    """
    Build local seed-clouds by sampling inside the Stage 1 box region around each kept seed.
    """

    def build_seed_clouds(
        self,
        seeds: List[SeedCandidate],
        config: SeedModeSearchConfig,
        query_context: Optional[Dict[str, Any]] = None,
    ) -> List[SeedCloud]:
        if len(seeds) == 0:
            return []

        device = seeds[0].coord_raw.device
        radius_raw = _resolve_stage1_radius_raw(
            config=config,
            query_context=query_context,
            device=device,
            round_index=0,
        )
        centers_raw = torch.stack([seed.coord_raw for seed in seeds], dim=0)
        n_samples = _resolve_stage1_sample_count(config, round_index=0)
        sample_coords = BatchedBoxSampler.sample_around_centers(
            centers_raw=centers_raw,
            radius_raw=radius_raw,
            n_samples=n_samples,
            method=config.seed_region.local_sample_method,
            query_context=query_context,
            enable_scale_sampling=bool(config.seed_region.enable_scale_sampling),
        )
        flat_coords = sample_coords.reshape(-1, 4)
        flat_scores = _score_raw_coords_chunked(
            flat_coords,
            query_context=query_context,
            score_key=_score_key_stage1(query_context),
            chunk_key="stage1_score_chunk_size",
        )
        sample_scores = flat_scores.reshape(len(seeds), n_samples)

        seed_clouds: List[SeedCloud] = []
        for seed, coords_seed, scores_seed in zip(seeds, sample_coords, sample_scores):
            seed_clouds.append(
                SeedCloud(
                    query_id=seed.query_id,
                    seed_id=seed.seed_id,
                    center_raw=seed.coord_raw.clone(),
                    radius_raw=radius_raw.clone(),
                    sample_coords_raw=coords_seed,
                    sample_scores=scores_seed,
                    metadata={"seed_score": seed.seed_score},
                )
            )
        return seed_clouds


class IterativeSeedCloudRefiner(BaseSeedCloudRelocator):
    """
    Repeatedly relocate, resample, and prune seed-clouds before turning them into modes.
    """

    def relocate_seed_clouds(
        self,
        seed_clouds: List[SeedCloud],
        config: SeedModeSearchConfig,
        query_contSeedModeSearchConfigext: Optional[Dict[str, Any]] = None,
    ) -> List[ModeState]:
        if len(seed_clouds) == 0:
            return []

        reverse = config.metric_goal == "maximize"
        active_clouds = list(seed_clouds)
        n_rounds = _resolve_stage1_round_count(config)
        round_summaries: Dict[str, Dict[str, Any]] = {}

        for round_index in range(n_rounds):
            cloud_infos: List[Dict[str, Any]] = []
            for cloud in active_clouds:
                coords = cloud.sample_coords_raw
                scores = cloud.sample_scores.reshape(-1)
                if coords.shape[0] == 0:
                    continue

                elite_count = max(1, int(math.ceil(coords.shape[0] * float(config.seed_region.elite_ratio))))
                elite_idx = torch.argsort(scores, descending=reverse)[:elite_count]
                elite_coords = coords.index_select(0, elite_idx)
                elite_scores = scores.index_select(0, elite_idx)

                weight_mode = str(config.seed_region.relocate_weight_mode).strip().lower()
                if weight_mode == "softmax":
                    logits = elite_scores - elite_scores.max()
                    elite_weights = torch.softmax(float(config.seed_region.relocate_weight_beta) * logits, dim=0)
                elif weight_mode == "linear":
                    elite_weights = _normalize_weights(elite_scores - elite_scores.min() + 1e-6)
                else:
                    raise ValueError(f"Unsupported relocate_weight_mode: {config.seed_region.relocate_weight_mode}")

                center_raw = _compute_weighted_center_raw(elite_coords, elite_weights)
                center_raw = _project_raw_coords(center_raw.reshape(1, 4), query_context=query_context).reshape(4)
                if not bool(config.seed_region.enable_scale_sampling):
                    center_raw[3] = cloud.center_raw[3]
                sigma_diag = _compute_sigma_diag_raw(elite_coords, center_raw, weights=elite_weights)
                best_idx = torch.argmax(scores) if reverse else torch.argmin(scores)
                best_coord = coords[best_idx].clone()
                best_score = float(scores[best_idx].item())
                selection_metric = float(elite_scores.mean().item())
                info = {
                    "cloud": cloud,
                    "center_raw": center_raw,
                    "sigma_diag": sigma_diag,
                    "best_coord": best_coord,
                    "best_score": best_score,
                    "selection_metric": selection_metric,
                    "elite_count": int(elite_count),
                }
                round_summaries[cloud.seed_id] = info
                cloud_infos.append(info)

            if len(cloud_infos) == 0:
                return []

            if round_index + 1 >= n_rounds:
                active_clouds = [info["cloud"] for info in cloud_infos]
                break

            keep_count = max(
                int(config.seed_region.min_surviving_clouds),
                int(math.ceil(len(cloud_infos) * _resolve_stage1_survival_ratio(config, round_index))),
            )
            keep_count = min(keep_count, len(cloud_infos))
            cloud_infos = sorted(
                cloud_infos,
                key=lambda item: float(item["selection_metric"]),
                reverse=reverse,
            )[:keep_count]

            next_clouds: List[SeedCloud] = []
            next_round_index = round_index + 1
            centers_next = torch.stack([info["center_raw"] for info in cloud_infos], dim=0)
            device = centers_next.device
            radius_raw = _resolve_stage1_radius_raw(
                config=config,
                query_context=query_context,
                device=device,
                round_index=next_round_index,
            )
            n_samples = _resolve_stage1_sample_count(config, round_index=next_round_index)
            coords_next_all = BatchedBoxSampler.sample_around_centers(
                centers_raw=centers_next,
                radius_raw=radius_raw,
                n_samples=n_samples,
                method=config.seed_region.local_sample_method,
                query_context=query_context,
                enable_scale_sampling=bool(config.seed_region.enable_scale_sampling),
            )
            scores_next_all = _score_raw_coords_chunked(
                coords_next_all.reshape(-1, 4),
                query_context=query_context,
                score_key=_score_key_stage1(query_context),
                chunk_key="stage1_score_chunk_size",
            ).reshape(len(cloud_infos), n_samples)

            for info, coords_next, scores_next in zip(cloud_infos, coords_next_all, scores_next_all):
                next_clouds.append(
                    SeedCloud(
                        query_id=info["cloud"].query_id,
                        seed_id=info["cloud"].seed_id,
                        center_raw=info["center_raw"].clone(),
                        radius_raw=radius_raw.clone(),
                        sample_coords_raw=coords_next,
                        sample_scores=scores_next,
                        metadata={
                            **info["cloud"].metadata,
                            "stage1_round_index": int(next_round_index),
                            "selection_metric": float(info["selection_metric"]),
                        },
                    )
                )
            active_clouds = next_clouds

        modes: List[ModeState] = []
        for idx, cloud in enumerate(active_clouds):
            info = round_summaries.get(cloud.seed_id, None)
            if info is None:
                continue
            center_raw = info["center_raw"].clone()
            sigma_diag = torch.maximum(info["sigma_diag"], cloud.radius_raw * 0.5)
            if not bool(config.seed_region.enable_scale_sampling):
                sigma_diag = sigma_diag.clone()
                sigma_diag[3] = 0.0
            mode_state = ModeState(
                query_id=cloud.query_id,
                mode_id=f"q{cloud.query_id}_mode{idx}",
                center_raw=center_raw,
                sigma_diag_raw=sigma_diag,
                score_mode=config.score_mode_stage3,
                stage="stage1.5",
                best_coord_raw=info["best_coord"].clone(),
                best_score=float(info["best_score"]),
                latest_metric=float(info["selection_metric"]),
                metadata={
                    "source_seed_id": cloud.seed_id,
                    "selection_metric": float(info["selection_metric"]),
                    "seed_score": cloud.metadata.get("seed_score", None),
                    "radius_raw": cloud.radius_raw.clone(),
                    "stage1_round_index": int(cloud.metadata.get("stage1_round_index", n_rounds - 1)),
                },
            )
            mode_state.record_event(
                "seed_cloud_refined",
                elite_count=int(info["elite_count"]),
                selection_metric=float(info["selection_metric"]),
                best_score=float(info["best_score"]),
            )
            modes.append(mode_state)
        return modes


class BatchedMultiStartEvolutionSeedCloudRefiner(BaseSeedCloudRelocator):
    """
    Batched multi-start evolution strategy for Stage 1.5.

    It treats each L0 seed as one local population, scores all active
    populations in one flattened tensor per round, and keeps each mode's
    historical best coord/score. It does not estimate or update covariance.
    """

    @staticmethod
    def _resolve_standard(config: SeedModeSearchConfig, key: str, default: str) -> str:
        value = str(config.metadata.get(key, default)).strip().lower()
        if value not in ("best", "elite_sum"):
            raise ValueError(f"{key} must be 'best' or 'elite_sum', got {value}")
        return value

    @staticmethod
    def _select_mode_list(values: List[Any], indices: torch.Tensor) -> List[Any]:
        return [values[int(idx)] for idx in indices.detach().cpu().tolist()]

    def relocate_seed_clouds(
        self,
        seed_clouds: List[SeedCloud],
        config: SeedModeSearchConfig,
        query_context: Optional[Dict[str, Any]] = None,
    ) -> List[ModeState]:
        if len(seed_clouds) == 0:
            return []

        reverse = config.metric_goal == "maximize"
        survive_stand = self._resolve_standard(config, "stage1_survive_stand", "best")
        move_stand = self._resolve_standard(config, "stage1_move_stand", "elite_sum")
        n_rounds = _resolve_stage1_round_count(config)

        device = seed_clouds[0].center_raw.device
        source_seed_ids = [cloud.seed_id for cloud in seed_clouds]
        seed_scores = [cloud.metadata.get("seed_score", None) for cloud in seed_clouds]
        centers = torch.stack([cloud.center_raw.to(device=device, dtype=torch.float32) for cloud in seed_clouds], dim=0)
        coords_round = torch.stack([cloud.sample_coords_raw.to(device=device, dtype=torch.float32) for cloud in seed_clouds], dim=0)
        scores_round = torch.stack([cloud.sample_scores.to(device=device, dtype=torch.float32).reshape(-1) for cloud in seed_clouds], dim=0)
        radius_round = torch.stack([cloud.radius_raw.to(device=device, dtype=torch.float32) for cloud in seed_clouds], dim=0)

        n_modes = int(coords_round.shape[0])
        if reverse:
            history_best_scores = torch.full((n_modes,), -1e9, device=device, dtype=torch.float32)
        else:
            history_best_scores = torch.full((n_modes,), 1e9, device=device, dtype=torch.float32)
        history_best_coords = centers.clone()
        history_best_round = torch.full((n_modes,), -1, device=device, dtype=torch.long)
        latest_selection_metric = history_best_scores.clone()
        latest_elite_metric = history_best_scores.clone()
        latest_elite_count = 0
        final_round_index = 0

        for round_index in range(n_rounds):
            final_round_index = int(round_index)
            if int(coords_round.shape[0]) == 0:
                return []
            if int(coords_round.shape[1]) == 0:
                return []

            n_active = int(coords_round.shape[0])
            n_samples = int(coords_round.shape[1])
            if n_samples <= 0:
                return []

            elite_count = max(1, int(math.ceil(n_samples * float(config.seed_region.elite_ratio))))
            elite_count = min(elite_count, n_samples)
            latest_elite_count = int(elite_count)

            if reverse:
                round_best_scores, round_best_idx = torch.max(scores_round, dim=1)
            else:
                round_best_scores, round_best_idx = torch.min(scores_round, dim=1)
            gather_idx = round_best_idx.reshape(n_active, 1, 1).expand(n_active, 1, 4)
            round_best_coords = torch.gather(coords_round, dim=1, index=gather_idx).squeeze(1)

            if reverse:
                improve_mask = round_best_scores > history_best_scores
            else:
                improve_mask = round_best_scores < history_best_scores
            history_best_scores = torch.where(improve_mask, round_best_scores, history_best_scores)
            history_best_coords = torch.where(improve_mask.reshape(-1, 1), round_best_coords, history_best_coords)
            history_best_round = torch.where(
                improve_mask,
                torch.full_like(history_best_round, int(round_index)),
                history_best_round,
            )

            elite_scores, elite_idx = torch.topk(scores_round, k=elite_count, dim=1, largest=reverse)
            elite_coords = torch.gather(
                coords_round,
                dim=1,
                index=elite_idx.unsqueeze(-1).expand(-1, -1, 4),
            )
            elite_weights = torch.ones(
                (n_active, elite_count),
                device=device,
                dtype=torch.float32,
            ) / max(1, elite_count)
            elite_centers = _compute_weighted_center_raw_batched(elite_coords, elite_weights)
            elite_centers = _project_raw_coords(elite_centers, query_context=query_context)
            if not bool(config.seed_region.enable_scale_sampling):
                elite_centers = elite_centers.clone()
                elite_centers[:, 3] = centers[:, 3]

            latest_elite_metric = elite_scores.sum(dim=1)
            if survive_stand == "best":
                latest_selection_metric = history_best_scores
            else:
                latest_selection_metric = latest_elite_metric

            if move_stand == "best":
                next_centers = history_best_coords
            else:
                next_centers = elite_centers

            if round_index + 1 >= n_rounds:
                centers = next_centers
                break

            keep_count = max(
                int(config.seed_region.min_surviving_clouds),
                int(math.ceil(n_active * _resolve_stage1_survival_ratio(config, round_index))),
            )
            keep_count = min(keep_count, n_active)
            _, keep_idx = torch.topk(latest_selection_metric, k=keep_count, largest=reverse)

            centers = next_centers.index_select(0, keep_idx)
            history_best_scores = history_best_scores.index_select(0, keep_idx)
            history_best_coords = history_best_coords.index_select(0, keep_idx)
            history_best_round = history_best_round.index_select(0, keep_idx)
            latest_selection_metric = latest_selection_metric.index_select(0, keep_idx)
            latest_elite_metric = latest_elite_metric.index_select(0, keep_idx)
            source_seed_ids = self._select_mode_list(source_seed_ids, keep_idx)
            seed_scores = self._select_mode_list(seed_scores, keep_idx)

            next_round_index = round_index + 1
            radius_next = _resolve_stage1_radius_raw(
                config=config,
                query_context=query_context,
                device=device,
                round_index=next_round_index,
            )
            n_samples_next = _resolve_stage1_sample_count(config, round_index=next_round_index)
            coords_round = BatchedBoxSampler.sample_around_centers(
                centers_raw=centers,
                radius_raw=radius_next,
                n_samples=n_samples_next,
                method=config.seed_region.local_sample_method,
                query_context=query_context,
                enable_scale_sampling=bool(config.seed_region.enable_scale_sampling),
            )
            scores_round = _score_raw_coords_chunked(
                coords_round.reshape(-1, 4),
                query_context=query_context,
                score_key=_score_key_stage1(query_context),
                chunk_key="stage1_score_chunk_size",
            ).reshape(int(coords_round.shape[0]), n_samples_next)
            radius_round = radius_next.reshape(1, 4).expand(int(coords_round.shape[0]), 4).clone()

        modes: List[ModeState] = []
        for idx in range(int(centers.shape[0])):
            sigma_diag = radius_round[idx].clone()
            if not bool(config.seed_region.enable_scale_sampling):
                sigma_diag[3] = 0.0
            best_score = float(history_best_scores[idx].item())
            selection_metric = float(latest_selection_metric[idx].item())
            mode_state = ModeState(
                query_id=seed_clouds[0].query_id,
                mode_id=f"q{seed_clouds[0].query_id}_mode{idx}",
                center_raw=centers[idx].clone(),
                sigma_diag_raw=sigma_diag,
                score_mode=config.score_mode_stage3,
                stage="stage1.5",
                best_coord_raw=history_best_coords[idx].clone(),
                best_score=best_score,
                latest_metric=selection_metric,
                metadata={
                    "source_seed_id": source_seed_ids[idx],
                    "selection_metric": selection_metric,
                    "seed_score": seed_scores[idx],
                    "radius_raw": radius_round[idx].clone(),
                    "stage1_round_index": int(final_round_index),
                    "stage1_5_refiner_mode": "multi_start_es",
                    "stage1_survive_stand": survive_stand,
                    "stage1_move_stand": move_stand,
                    "history_best_round": int(history_best_round[idx].item()),
                    "history_best_score": best_score,
                    "elite_score_sum": float(latest_elite_metric[idx].item()),
                    "elite_count": int(latest_elite_count),
                    "n_rounds": int(n_rounds),
                },
            )
            mode_state.record_event(
                "multi_start_es_refined",
                elite_count=int(latest_elite_count),
                selection_metric=selection_metric,
                best_score=best_score,
                survive_stand=survive_stand,
                move_stand=move_stand,
            )
            modes.append(mode_state)
        return modes


BatchedFixedStepEvolutionSeedCloudRefiner = BatchedMultiStartEvolutionSeedCloudRefiner


class PassthroughModeDeduper(BaseModeDeduper):
    """
    Keep all relocated modes. Stage 1.5 no longer applies threshold-based dedup.
    """

    def dedup_modes(
        self,
        modes_init: List[ModeState],
        config: SeedModeSearchConfig,
    ) -> List[ModeState]:
        for mode in modes_init:
            mode.alive = True
            mode.stage = "stage1.5"
            mode.record_event("stage1_5_passthrough")
        return modes_init


class EvoTorchFinalModeOptimizer(BaseFinalModeOptimizer):
    """
    Stage 3 adapter which reuses the existing EvoTorch multi-start refiner.

    The first version runs one CMA search per mode so each mode can keep its own
    Stage-2-derived initialization sigma.
    """

    def optimize_modes(
        self,
        query_id: int,
        modes: List[ModeState],
        config: SeedModeSearchConfig,
        query_context: Optional[Dict[str, Any]] = None,
    ) -> List[ModeState]:
        if len(modes) == 0 or not bool(config.stage3.enable):
            return modes

        if query_context is None or "query_feat" not in query_context:
            raise KeyError("query_context must provide `query_feat` for Stage 3 CMA optimization.")
        if query_context is None or "score_pair_fn" not in query_context:
            raise KeyError("query_context must provide `score_pair_fn` for Stage 3 CMA optimization.")

        query_feat = _as_float_tensor(query_context["query_feat"]).reshape(1, -1)
        score_pair_fn = query_context["score_pair_fn"]
        cma_refiner = query_context.get("cma_refiner", None)
        if cma_refiner is None:
            from trainers.util_stage3_multi_start_CMAES_by_evotorch import MultiStartCMAESEvoTorchRefiner

            if "coords_processor" not in query_context:
                raise KeyError("query_context must provide `coords_processor` to construct the CMA refiner.")
            cma_refiner = MultiStartCMAESEvoTorchRefiner(
                coords_processor=query_context["coords_processor"],
                device=query_feat.device,
            )

        reverse = config.metric_goal == "maximize"
        ranked_modes = sorted(
            modes,
            key=lambda m: float(m.best_score if m.best_score is not None else (m.latest_metric or -1e9)),
            reverse=reverse,
        )
        selected_modes = ranked_modes[: min(int(config.stage3.cma_max_input_mode), len(ranked_modes))]
        if len(selected_modes) == 0:
            return []

        device = selected_modes[0].center_raw.device
        verbose = bool(config.metadata.get("verbose_stage_timing", False))
        total_iters = max(1, int(config.stage3.n_iterations))
        competition_enabled = bool(config.stage3.enable_competition)
        warmup_iters = min(total_iters, max(1, int(config.stage3.early_stop_patience)))
        interval_iters = max(1, int(config.stage3.competition_interval))
        optimize_dims = (0, 1, 2, 3) if bool(config.stage3.cma_optimize_scale) else (0, 1, 2)

        active_modes = [mode.clone() for mode in selected_modes]
        for mode_state in active_modes:
            mode_state.metadata["pre_stage3_best_score"] = mode_state.best_score
            mode_state.metadata["pre_stage3_best_coord_raw"] = (
                None if mode_state.best_coord_raw is None else mode_state.best_coord_raw.clone()
            )
            mode_state.best_score = None
            mode_state.best_coord_raw = mode_state.center_raw.clone()

        sigma0_active = torch.tensor(
            [_resolve_stage3_init_sigma(mode, config=config, query_context=query_context) for mode in active_modes],
            device=device,
            dtype=torch.float32,
        )
        iters_done = 0
        chunk_index = 0

        while len(active_modes) > 0 and iters_done < total_iters:
            if competition_enabled and iters_done == 0:
                n_chunk_iters = warmup_iters
            elif competition_enabled:
                n_chunk_iters = min(interval_iters, total_iters - iters_done)
            else:
                n_chunk_iters = total_iters - iters_done

            seed_coords_batch = torch.stack([mode.center_raw for mode in active_modes], dim=0).reshape(1, -1, 4)
            sigma0_batch = sigma0_active.reshape(1, -1)
            refine_result = cma_refiner.refine_batch_queries(
                query_feats=query_feat,
                seed_coords_raw_batch=seed_coords_batch,
                score_pair_fn=score_pair_fn,
                sigma0=sigma0_batch,
                popsize=int(config.stage3.popsize),
                n_iterations=int(n_chunk_iters),
                maximize=reverse,
                cma_seed=int(chunk_index * 10000),
                cma_variant=config.stage3.cma_variant,
                enable_early_stop=bool(config.stage3.enable_early_stop),
                early_stop_patience=int(config.stage3.early_stop_patience),
                return_diagnostics=True,
                optimize_dims=optimize_dims,
            )
            coords_best = _as_float_tensor(refine_result["best_coords"], device=device).reshape(-1, 4)
            scores_best = _as_float_tensor(refine_result["best_scores"], device=device).reshape(-1)
            final_population_scores = _as_float_tensor(
                refine_result["final_population_scores"],
                device=device,
            ).reshape(len(active_modes), -1)

            for mode_idx, mode_state in enumerate(active_modes):
                score_new = float(scores_best[mode_idx].item())
                coord_new = coords_best[mode_idx].clone()
                score_prev = mode_state.best_score
                if score_prev is None:
                    is_better = True
                else:
                    is_better = score_new >= score_prev if reverse else score_new <= score_prev
                if is_better:
                    mode_state.best_score = score_new
                    mode_state.best_coord_raw = coord_new
                mode_state.center_raw = coord_new
                mode_state.latest_metric = score_new
                mode_state.stage = "stage3"
                mode_state.record_event(
                    "stage3_cma_chunk",
                    chunk_index=int(chunk_index),
                    n_chunk_iters=int(n_chunk_iters),
                    sigma0=float(sigma0_active[mode_idx].item()),
                    best_score=float(mode_state.best_score if mode_state.best_score is not None else score_new),
                )

            iters_done += int(n_chunk_iters)
            chunk_index += 1

            if verbose:
                print(
                    f"[SeedMode][Q{query_id}] Stage3-Chunk{chunk_index - 1} done | "
                    f"n_mode={len(active_modes)} iters={n_chunk_iters} total_iters={iters_done}/{total_iters}"
                )

            if not competition_enabled or iters_done >= total_iters or len(active_modes) <= int(config.stage3.min_surviving_modes):
                continue

            metric_list = [
                _compute_weighted_elite_metric(
                    scores=final_population_scores[mode_idx],
                    elite_ratio=float(config.stage3.elite_ratio),
                    metric_goal=config.metric_goal,
                    beta=float(config.stage3.metric_weight_beta),
                )
                for mode_idx in range(len(active_modes))
            ]
            rank_idx = sorted(
                range(len(active_modes)),
                key=lambda idx: float(metric_list[idx]),
                reverse=reverse,
            )
            keep_count = max(
                int(config.stage3.min_surviving_modes),
                int(math.ceil(len(active_modes) * float(config.stage3.survival_ratio))),
            )
            keep_count = min(keep_count, len(active_modes))
            keep_idx = rank_idx[:keep_count]

            if verbose:
                print(
                    f"[SeedMode][Q{query_id}] Stage3-Prune after chunk{chunk_index - 1} | "
                    f"n_in={len(active_modes)} n_keep={len(keep_idx)}"
                )

            next_modes: List[ModeState] = []
            next_sigma0: List[float] = []
            for idx in keep_idx:
                mode_state = active_modes[idx]
                mode_state.latest_metric = float(metric_list[idx])
                mode_state.record_event(
                    "stage3_prune_keep",
                    chunk_index=int(chunk_index - 1),
                    selection_metric=float(metric_list[idx]),
                )
                next_modes.append(mode_state)
                next_sigma0.append(float(sigma0_active[idx].item()))

            active_modes = next_modes
            sigma0_active = torch.tensor(next_sigma0, device=device, dtype=torch.float32)

        if bool(config.stage3.rerank_per_mode_after_stage3) and len(active_modes) > 0:
            with torch.no_grad():
                reuse_saved_scores = str(config.score_mode_stage1).lower() == str(config.stage3.score_mode).lower()
                for mode_state in active_modes:
                    stage3_best_coord = None if mode_state.best_coord_raw is None else mode_state.best_coord_raw.clone()
                    stage3_best_score = None if mode_state.best_score is None else float(mode_state.best_score)
                    pre_stage3_best_coord = mode_state.metadata.get("pre_stage3_best_coord_raw", None)
                    pre_stage3_best_score = mode_state.metadata.get("pre_stage3_best_score", None)

                    candidate_names = []
                    candidate_coords = []
                    candidate_scores = []
                    if stage3_best_coord is not None:
                        candidate_names.append("stage3_best")
                        candidate_coords.append(stage3_best_coord)
                        candidate_scores.append(stage3_best_score)
                    if pre_stage3_best_coord is not None:
                        candidate_names.append("stage1_5_best")
                        candidate_coords.append(pre_stage3_best_coord.clone())
                        candidate_scores.append(
                            None if pre_stage3_best_score is None else float(pre_stage3_best_score)
                        )

                    if len(candidate_coords) == 0:
                        continue

                    if reuse_saved_scores and all(score is not None for score in candidate_scores):
                        scores_candidates = torch.tensor(candidate_scores, device=device, dtype=torch.float32)
                        rerank_mode = "reuse_saved_scores"
                    else:
                        coords_candidates = torch.stack(candidate_coords, dim=0).to(device=device, dtype=torch.float32)
                        query_feat_candidates = query_feat.expand(coords_candidates.shape[0], -1)
                        scores_candidates = _as_float_tensor(
                            score_pair_fn(query_feat_candidates, coords_candidates),
                            device=device,
                        ).reshape(-1)
                        rerank_mode = "rescore_with_stage3"

                    winner_idx = torch.argmax(scores_candidates) if reverse else torch.argmin(scores_candidates)
                    winner_idx_int = int(winner_idx.item())
                    winner_coord = candidate_coords[winner_idx_int].clone().to(device=device, dtype=torch.float32)
                    winner_score = float(scores_candidates[winner_idx_int].item())

                    mode_state.metadata["stage3_best_coord_raw_before_rerank"] = (
                        None if stage3_best_coord is None else stage3_best_coord.clone()
                    )
                    mode_state.metadata["stage3_best_score_before_rerank"] = stage3_best_score
                    mode_state.metadata["stage3_rerank_winner"] = candidate_names[winner_idx_int]
                    mode_state.metadata["stage3_rerank_mode"] = rerank_mode
                    mode_state.metadata["stage3_rerank_scores"] = {
                        name: float(score)
                        for name, score in zip(candidate_names, scores_candidates.detach().cpu().tolist())
                    }

                    mode_state.best_coord_raw = winner_coord
                    mode_state.best_score = winner_score
                    mode_state.latest_metric = winner_score
                    mode_state.center_raw = winner_coord.clone()
                    mode_state.record_event(
                        "stage3_rerank_select",
                        winner=candidate_names[winner_idx_int],
                        winner_score=winner_score,
                        n_candidates=len(candidate_names),
                    )

            if verbose:
                print(f"[SeedMode][Q{query_id}] Stage3-Rerank done | n_mode={len(active_modes)}")

        return active_modes


class GradientTopKOptimizer(BaseStage4Optimizer):
    """
    Gradient-based Stage 4 optimizer over the latest stage trace record.
    """

    @staticmethod
    def _select_input_record(
        stage_trace: QueryStageTrace,
        config: SeedModeSearchConfig,
    ) -> Optional[StageTopKRecord]:
        input_stage = str(config.stage4.input_stage).strip().lower()
        if input_stage == "latest":
            return stage_trace.latest_record()

        candidates = [record for record in stage_trace.stage_records if record.stage_name.lower() == input_stage]
        if len(candidates) == 0:
            return None
        return max(candidates, key=lambda record: int(record.stage_id))

    @staticmethod
    def _opt_coords_topk(
        coords_topk: torch.Tensor,
        feat_q: torch.Tensor,
        config: SeedModeSearchConfig,
        query_context: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        feat_q = feat_q.detach().to(coords_topk.device)
        coords_processor = _require_query_context_value(query_context, "coords_processor")
        get_feats_fm_grid_fn = _require_query_context_value(query_context, "get_feats_fm_grid_fn")
        grid_module = _require_query_context_value(query_context, "grid_module")
        grid_mlp_module = _require_query_context_value(query_context, "grid_mlp_module")
        pos_encoder_grid_module = _require_query_context_value(query_context, "pos_encoder_grid_module")

        if coords_topk.ndim != 3:
            raise ValueError(f"coords_topk must be [B,K,4], got {tuple(coords_topk.shape)}")
        batch_size, topk, _ = coords_topk.shape
        feat_dim = feat_q.shape[-1]
        feat_q_expanded = feat_q.unsqueeze(1).expand(batch_size, topk, feat_dim).reshape(-1, feat_dim)
        coords_init_flat = coords_topk.reshape(-1, 4).clone().detach()

        grid_was_training = grid_module.training
        grid_mlp_was_training = grid_mlp_module.training
        pos_encoder_was_training = pos_encoder_grid_module.training
        grid_module.train()
        grid_mlp_module.train()
        pos_encoder_grid_module.train()

        optimize_scale = bool(config.stage4.optimize_scale)
        xy_param = coords_init_flat[:, :2].clone().requires_grad_(True)
        rot_param = coords_init_flat[:, 2:3].clone().requires_grad_(True)
        scale_fixed = coords_init_flat[:, 3:4].clone().detach()
        if optimize_scale:
            scale_param = coords_init_flat[:, 3:4].clone().requires_grad_(True)
            param_groups = [
                {"params": [xy_param], "lr": float(config.stage4.lr_xy)},
                {"params": [rot_param], "lr": float(config.stage4.lr_rot)},
                {"params": [scale_param], "lr": float(config.stage4.lr_scale)},
            ]
            grad_params = [xy_param, rot_param, scale_param]
        else:
            scale_param = scale_fixed
            param_groups = [
                {"params": [xy_param], "lr": float(config.stage4.lr_xy)},
                {"params": [rot_param], "lr": float(config.stage4.lr_rot)},
            ]
            grad_params = [xy_param, rot_param]

        optimizer = torch.optim.Adam(param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(config.stage4.n_steps)),
            eta_min=1e-6,
        )
        log_interval = max(1, int(config.stage4.log_interval))

        for step_idx in range(int(config.stage4.n_steps)):
            optimizer.zero_grad()
            with torch.enable_grad():
                coords2opt_free = torch.cat([xy_param, rot_param, scale_param], dim=-1)
                coords2opt = _project_raw_coords(coords2opt_free, query_context=query_context)
                if not optimize_scale:
                    coords2opt = torch.cat([coords2opt[:, :3], scale_fixed], dim=-1)

                coords_6d = coords_processor.raw_to_net(coords2opt, append_linear_rot=True)
                grid_input = torch.cat([coords_6d[:, :2], coords_6d[:, -1:]], dim=-1)
                feat_ref_raw = get_feats_fm_grid_fn(grid_input)
                coords_encoded = pos_encoder_grid_module(coords_6d[:, :5])
                feat_ref = grid_mlp_module(feat_ref_raw, coords_encoded)
                feat_ref = TF.normalize(feat_ref, dim=-1, p=2)
                loss_vec = TF.mse_loss(feat_q_expanded, feat_ref, reduction="none").mean(dim=-1)
                loss_mean = loss_vec.mean()
                loss = loss_vec.sum()
                loss.backward()

            torch.nn.utils.clip_grad_norm_(grad_params, max_norm=float(config.stage4.max_grad_norm))
            optimizer.step()
            scheduler.step()

            if bool(config.stage4.verbose) and (step_idx % log_interval == 0 or step_idx == int(config.stage4.n_steps) - 1):
                with torch.no_grad():
                    coords_log = torch.cat([xy_param, rot_param, scale_param], dim=-1)
                    coords_log = _project_raw_coords(coords_log, query_context=query_context)
                    if not optimize_scale:
                        coords_log = torch.cat([coords_log[:, :3], scale_fixed], dim=-1)
                    score_mean, score_top1 = _summarize_stage4_scores(
                        coords_raw=coords_log,
                        config=config,
                        query_context=query_context,
                    )
                print(
                    f"[Stage4-Grad] batch={batch_size} topk={topk} "
                    f"step={step_idx + 1}/{int(config.stage4.n_steps)} "
                    f"loss_mean={float(loss_mean.item()):.6f} "
                    f"score_mean={score_mean:.6f} score_top1={score_top1:.6f}"
                )

        with torch.no_grad():
            coords_final_flat = torch.cat([xy_param, rot_param, scale_param], dim=-1)
            coords_final_flat = _project_raw_coords(coords_final_flat, query_context=query_context)
            if not optimize_scale:
                coords_final_flat = torch.cat([coords_final_flat[:, :3], scale_fixed], dim=-1)

            coords_6d_final = coords_processor.raw_to_net(coords_final_flat, append_linear_rot=True)
            grid_input_final = torch.cat([coords_6d_final[:, :2], coords_6d_final[:, -1:]], dim=-1)
            feat_ref_final_raw = get_feats_fm_grid_fn(grid_input_final)
            coords_encoded_final = pos_encoder_grid_module(coords_6d_final[:, :5])
            feat_ref_final = grid_mlp_module(feat_ref_final_raw, coords_encoded_final)
            feat_ref_final = TF.normalize(feat_ref_final, dim=-1, p=2)
            final_loss_flat = TF.mse_loss(feat_q_expanded, feat_ref_final, reduction="none").mean(dim=-1)

        if not grid_was_training:
            grid_module.eval()
        if not grid_mlp_was_training:
            grid_mlp_module.eval()
        if not pos_encoder_was_training:
            pos_encoder_grid_module.eval()

        coords_opted_topk_t = coords_final_flat.reshape(batch_size, topk, 4)
        return coords_opted_topk_t

    @staticmethod
    def _opt_coords_topk_linear4d(
        coords_topk: torch.Tensor,
        feat_q: torch.Tensor,
        config: SeedModeSearchConfig,
        query_context: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        feat_q = feat_q.detach().to(coords_topk.device)
        coords_processor = _require_query_context_value(query_context, "coords_processor")
        get_feats_fm_grid_fn = _require_query_context_value(query_context, "get_feats_fm_grid_fn")
        grid_module = _require_query_context_value(query_context, "grid_module")
        grid_mlp_module = _require_query_context_value(query_context, "grid_mlp_module")
        pos_encoder_grid_module = _require_query_context_value(query_context, "pos_encoder_grid_module")

        if coords_topk.ndim != 3:
            raise ValueError(f"coords_topk must be [B,K,4], got {tuple(coords_topk.shape)}")
        batch_size, topk, _ = coords_topk.shape
        feat_dim = feat_q.shape[-1]
        feat_q_expanded = feat_q.unsqueeze(1).expand(batch_size, topk, feat_dim).reshape(-1, feat_dim)
        coords_init_raw_flat = coords_topk.reshape(-1, 4).clone().detach()
        coords_init_linear_flat = coords_processor.raw_to_linear(coords_init_raw_flat)

        grid_was_training = grid_module.training
        grid_mlp_was_training = grid_mlp_module.training
        pos_encoder_was_training = pos_encoder_grid_module.training
        grid_module.train()
        grid_mlp_module.train()
        pos_encoder_grid_module.train()

        optimize_scale = bool(config.stage4.optimize_scale)
        xy_param = coords_init_linear_flat[:, :2].clone().requires_grad_(True)
        rot_param = coords_init_linear_flat[:, 2:3].clone().requires_grad_(True)
        scale_fixed = coords_init_linear_flat[:, 3:4].clone().detach()
        if optimize_scale:
            scale_param = coords_init_linear_flat[:, 3:4].clone().requires_grad_(True)
            param_groups = [
                {"params": [xy_param], "lr": float(config.stage4.lr_xy)},
                {"params": [rot_param], "lr": float(config.stage4.lr_rot)},
                {"params": [scale_param], "lr": float(config.stage4.lr_scale)},
            ]
            grad_params = [xy_param, rot_param, scale_param]
        else:
            scale_param = scale_fixed
            param_groups = [
                {"params": [xy_param], "lr": float(config.stage4.lr_xy)},
                {"params": [rot_param], "lr": float(config.stage4.lr_rot)},
            ]
            grad_params = [xy_param, rot_param]

        optimizer = torch.optim.Adam(param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(config.stage4.n_steps)),
            eta_min=1e-6,
        )
        log_interval = max(1, int(config.stage4.log_interval))

        for step_idx in range(int(config.stage4.n_steps)):
            optimizer.zero_grad()
            with torch.enable_grad():
                coords_linear_free = torch.cat([xy_param, rot_param, scale_param], dim=-1)
                coords_linear = _project_linear_coords(coords_linear_free)
                if not optimize_scale:
                    coords_linear = torch.cat([coords_linear[:, :3], scale_fixed], dim=-1)

                coords_net5 = coords_processor.linear_to_net(coords_linear)
                grid_input = torch.cat([coords_linear[:, :2], coords_linear[:, 2:3]], dim=-1)
                feat_ref_raw = get_feats_fm_grid_fn(grid_input)
                coords_encoded = pos_encoder_grid_module(coords_net5[:, :5])
                feat_ref = grid_mlp_module(feat_ref_raw, coords_encoded)
                feat_ref = TF.normalize(feat_ref, dim=-1, p=2)
                loss_vec = TF.mse_loss(feat_q_expanded, feat_ref, reduction="none").mean(dim=-1)
                loss_mean = loss_vec.mean()
                loss = loss_vec.sum()
                loss.backward()

            torch.nn.utils.clip_grad_norm_(grad_params, max_norm=float(config.stage4.max_grad_norm))
            optimizer.step()
            scheduler.step()

            if bool(config.stage4.verbose) and (step_idx % log_interval == 0 or step_idx == int(config.stage4.n_steps) - 1):
                with torch.no_grad():
                    coords_linear_log = torch.cat([xy_param, rot_param, scale_param], dim=-1)
                    coords_linear_log = _project_linear_coords(coords_linear_log)
                    if not optimize_scale:
                        coords_linear_log = torch.cat([coords_linear_log[:, :3], scale_fixed], dim=-1)
                    coords_raw_log = coords_processor.linear_to_raw(coords_linear_log)
                    score_mean, score_top1 = _summarize_stage4_scores(
                        coords_raw=coords_raw_log,
                        config=config,
                        query_context=query_context,
                    )
                print(
                    f"[Stage4-Grad-Linear] batch={batch_size} topk={topk} "
                    f"step={step_idx + 1}/{int(config.stage4.n_steps)} "
                    f"loss_mean={float(loss_mean.item()):.6f} "
                    f"score_mean={score_mean:.6f} score_top1={score_top1:.6f}"
                )

        with torch.no_grad():
            coords_linear_final = torch.cat([xy_param, rot_param, scale_param], dim=-1)
            coords_linear_final = _project_linear_coords(coords_linear_final)
            if not optimize_scale:
                coords_linear_final = torch.cat([coords_linear_final[:, :3], scale_fixed], dim=-1)

            coords_net5_final = coords_processor.linear_to_net(coords_linear_final)
            grid_input_final = torch.cat([coords_linear_final[:, :2], coords_linear_final[:, 2:3]], dim=-1)
            feat_ref_final_raw = get_feats_fm_grid_fn(grid_input_final)
            coords_encoded_final = pos_encoder_grid_module(coords_net5_final[:, :5])
            feat_ref_final = grid_mlp_module(feat_ref_final_raw, coords_encoded_final)
            feat_ref_final = TF.normalize(feat_ref_final, dim=-1, p=2)
            final_loss_flat = TF.mse_loss(feat_q_expanded, feat_ref_final, reduction="none").mean(dim=-1)
            coords_final_flat = coords_processor.linear_to_raw(coords_linear_final)

        if not grid_was_training:
            grid_module.eval()
        if not grid_mlp_was_training:
            grid_mlp_module.eval()
        if not pos_encoder_was_training:
            pos_encoder_grid_module.eval()

        coords_opted_topk_t = coords_final_flat.reshape(batch_size, topk, 4)
        return coords_opted_topk_t

    def optimize_from_trace(
        self,
        query_id: int,
        stage_trace: QueryStageTrace,
        config: SeedModeSearchConfig,
        query_context: Optional[Dict[str, Any]] = None,
    ) -> List[ModeState]:
        input_record = self._select_input_record(stage_trace=stage_trace, config=config)
        if input_record is None or int(input_record.coords_topk_raw.shape[0]) == 0:
            return []

        topk = min(int(config.stage4.topk_input), int(input_record.coords_topk_raw.shape[0]))
        coords_input = input_record.coords_topk_raw[:topk].to(dtype=torch.float32)
        if coords_input.ndim != 2:
            raise ValueError(f"Stage4 expects [K,4] coords, got {tuple(coords_input.shape)}")
        coords_input_flat = coords_input.clone()
        coords_input = coords_input.unsqueeze(0)
        feat_q = _require_query_context_value(query_context, "query_feat")
        device = feat_q.device

        opt_space = str(config.stage4.opt_space).strip().lower()
        if opt_space == "linear":
            coords_opted = self._opt_coords_topk_linear4d(
                coords_topk=coords_input.to(device=device),
                feat_q=feat_q,
                config=config,
                query_context=query_context,
            ).reshape(-1, 4)
        elif opt_space == "raw":
            coords_opted = self._opt_coords_topk(
                coords_topk=coords_input.to(device=device),
                feat_q=feat_q,
                config=config,
                query_context=query_context,
            ).reshape(-1, 4)
        else:
            raise ValueError(f"Unsupported stage4 opt_space: {config.stage4.opt_space}")

        score_key = "score_coords_fn_stage4" if query_context is not None and "score_coords_fn_stage4" in query_context else "score_coords_fn"
        coords_before = coords_input_flat.to(device=device)
        scores_before = _score_raw_coords_chunked(
            coords_before,
            query_context=query_context,
            score_key=score_key,
            chunk_key="stage4_score_chunk_size",
        ).reshape(-1)
        scores_opted = _score_raw_coords_chunked(
            coords_opted,
            query_context=query_context,
            score_key=score_key,
            chunk_key="stage4_score_chunk_size",
        ).reshape(-1)
        reverse = config.metric_goal == "maximize"
        if reverse:
            use_opted = scores_opted >= scores_before
        else:
            use_opted = scores_opted <= scores_before
        coords_merged = torch.where(use_opted.reshape(-1, 1), coords_opted, coords_before)
        scores_merged = torch.where(use_opted, scores_opted, scores_before)
        order = torch.argsort(scores_merged, descending=reverse)

        modes_stage4: List[ModeState] = []
        for rank, idx in enumerate(order.tolist()):
            coord = coords_merged[idx].clone()
            score = float(scores_merged[idx].item())
            modes_stage4.append(
                ModeState(
                    query_id=query_id,
                    mode_id=f"q{query_id}_stage4_{rank}",
                    center_raw=coord.clone(),
                    sigma_diag_raw=torch.zeros((4,), device=device, dtype=torch.float32),
                    score_mode=config.stage4.score_mode,
                    stage="stage4",
                    best_coord_raw=coord.clone(),
                    best_score=score,
                    latest_metric=score,
                    metadata={
                        "source_stage": input_record.stage_name,
                        "source_stage_id": int(input_record.stage_id),
                        "source_mode_id": input_record.mode_ids[idx] if idx < len(input_record.mode_ids) else None,
                        "selected_variant": "opted" if bool(use_opted[idx].item()) else "input",
                        "score_before": float(scores_before[idx].item()),
                        "score_after": float(scores_opted[idx].item()),
                    },
                )
            )
        return modes_stage4


class SeedModeSearchPipeline:
    """
    Multi-stage search pipeline:

    Stage 1:
        seed screening -> seed-cloud construction
    Stage 1.5:
        seed-cloud relocation -> initial modes
    Stage 3:
        final local optimizer such as CMA-ES
    Stage 4:
        gradient-based local refinement
    """

    def __init__(
        self,
        seed_screening: BaseSeedScreeningStrategy,
        seed_cloud_builder: BaseSeedCloudBuilder,
        seed_cloud_relocator: BaseSeedCloudRelocator,
        mode_deduper: BaseModeDeduper,
        final_mode_optimizer: Optional[BaseFinalModeOptimizer] = None,
        stage4_optimizer: Optional[BaseStage4Optimizer] = None,
        config: Optional[SeedModeSearchConfig] = None,
    ):
        self.seed_screening = seed_screening
        self.seed_cloud_builder = seed_cloud_builder
        self.seed_cloud_relocator = seed_cloud_relocator
        self.mode_deduper = mode_deduper
        self.final_mode_optimizer = final_mode_optimizer
        self.stage4_optimizer = stage4_optimizer
        self.config = config or SeedModeSearchConfig()

    @staticmethod
    def _make_stage_record(
        query_id: int,
        stage_id: int,
        stage_name: str,
        score_func_name: str,
        modes: List[ModeState],
        topk: Optional[int] = None,
    ) -> StageTopKRecord:
        if len(modes) == 0:
            return StageTopKRecord(
                stage_id=int(stage_id),
                stage_name=stage_name,
                score_func_name=score_func_name,
                coords_topk_raw=torch.zeros((0, 4), dtype=torch.float32),
                scores_topk=torch.zeros((0,), dtype=torch.float32),
                metadata={"query_id": int(query_id)},
            )

        modes_sorted = sorted(
            modes,
            key=lambda m: float(m.best_score if m.best_score is not None else (m.latest_metric or -1e9)),
            reverse=True,
        )
        if topk is not None:
            modes_sorted = modes_sorted[: max(1, int(topk))]
        device = modes_sorted[0].center_raw.device
        coords_topk = torch.stack(
            [(m.best_coord_raw if m.best_coord_raw is not None else m.center_raw).to(device) for m in modes_sorted],
            dim=0,
        )
        scores_topk = torch.tensor(
            [float(m.best_score if m.best_score is not None else (m.latest_metric or -1e9)) for m in modes_sorted],
            device=device,
            dtype=torch.float32,
        )
        return StageTopKRecord(
            stage_id=int(stage_id),
            stage_name=stage_name,
            score_func_name=str(score_func_name),
            coords_topk_raw=coords_topk,
            scores_topk=scores_topk,
            mode_ids=[m.mode_id for m in modes_sorted],
            metadata={"query_id": int(query_id), "n_mode": len(modes_sorted)},
        )

    def run_query(
        self,
        query_id: int,
        coarse_seed_coords_raw: torch.Tensor,
        coarse_seed_scores: torch.Tensor,
        query_context: Optional[Dict[str, Any]] = None,
    ) -> SeedModeSearchResult:
        verbose = bool(self.config.metadata.get("verbose_stage_timing", False))
        t_query0 = time.perf_counter()

        t0 = time.perf_counter()
        seeds_kept = self.seed_screening.select_seeds(
            query_id=query_id,
            coarse_seed_coords_raw=coarse_seed_coords_raw,
            coarse_seed_scores=coarse_seed_scores,
            config=self.config,
            query_context=query_context,
        )
        t_seed = time.perf_counter() - t0
        if verbose:
            print(f"[SeedMode][Q{query_id}] Stage1-SeedScreen done | n_seed={len(seeds_kept)} | {t_seed:.3f}s")

        t0 = time.perf_counter()
        seed_clouds = self.seed_cloud_builder.build_seed_clouds(
            seeds=seeds_kept,
            config=self.config,
            query_context=query_context,
        )
        t_cloud = time.perf_counter() - t0
        if verbose:
            print(f"[SeedMode][Q{query_id}] Stage1-SeedCloud done | n_cloud={len(seed_clouds)} | {t_cloud:.3f}s")

        t0 = time.perf_counter()
        modes_relocated = self.seed_cloud_relocator.relocate_seed_clouds(
            seed_clouds=seed_clouds,
            config=self.config,
            query_context=query_context,
        )
        t_reloc = time.perf_counter() - t0
        if verbose:
            print(f"[SeedMode][Q{query_id}] Stage1.5-Relocate done | n_mode_raw={len(modes_relocated)} | {t_reloc:.3f}s")

        t0 = time.perf_counter()
        modes_init = self.mode_deduper.dedup_modes(
            modes_init=modes_relocated,
            config=self.config,
        )
        t_stage15_finalize = time.perf_counter() - t0
        if verbose:
            print(f"[SeedMode][Q{query_id}] Stage1.5-Finalize done | n_mode_init={len(modes_init)} | {t_stage15_finalize:.3f}s")

        stage_trace = QueryStageTrace(query_id=int(query_id))
        stage_trace.add_record(
            self._make_stage_record(
                query_id=query_id,
                stage_id=15,
                stage_name="stage1.5",
                score_func_name=self.config.score_mode_stage1,
                modes=modes_init,
                topk=None,
            )
        )

        active_modes = [mode.clone() for mode in modes_init if mode.alive]
        for mode_state in active_modes:
            mode_state.stage = "before_stage3"
            mode_state.record_event("before_stage3")
        if verbose:
            print(f"[SeedMode][Q{query_id}] Before-Stage3 ready | n_mode={len(active_modes)}")

        modes_before_stage3 = [mode.clone() for mode in active_modes]
        modes_final = modes_before_stage3
        if self.final_mode_optimizer is not None and bool(self.config.stage3.enable) and len(modes_before_stage3) > 0:
            t0 = time.perf_counter()
            modes_final = self.final_mode_optimizer.optimize_modes(
                query_id=query_id,
                modes=[mode.clone() for mode in modes_before_stage3],
                config=self.config,
                query_context=query_context,
            )
            t_stage3 = time.perf_counter() - t0
            if verbose:
                print(
                    f"[SeedMode][Q{query_id}] Stage3-CMA done | "
                    f"n_in={len(modes_before_stage3)} n_out={len(modes_final)} | {t_stage3:.3f}s"
                )
        elif verbose:
            print(f"[SeedMode][Q{query_id}] Stage3 skipped | n_mode={len(modes_before_stage3)}")

        if self.final_mode_optimizer is not None and len(modes_final) > 0:
            stage_trace.add_record(
                self._make_stage_record(
                    query_id=query_id,
                    stage_id=30,
                    stage_name="stage3",
                    score_func_name=self.config.score_mode_stage3,
                    modes=modes_final,
                    topk=None,
                )
            )

        if self.stage4_optimizer is not None and bool(self.config.stage4.enable):
            stage4_input_record = stage_trace.latest_record()
            t0 = time.perf_counter()
            modes_stage4 = self.stage4_optimizer.optimize_from_trace(
                query_id=query_id,
                stage_trace=stage_trace,
                config=self.config,
                query_context=query_context,
            )
            t_stage4 = time.perf_counter() - t0
            if len(modes_stage4) > 0:
                modes_final = modes_stage4
                stage_trace.add_record(
                    self._make_stage_record(
                        query_id=query_id,
                        stage_id=40,
                        stage_name="stage4",
                        score_func_name=self.config.stage4.score_mode,
                        modes=modes_final,
                        topk=None,
                    )
                )
            if verbose:
                print(
                    f"[SeedMode][Q{query_id}] Stage4-Grad done | "
                    f"n_in={int(stage4_input_record.coords_topk_raw.shape[0]) if stage4_input_record is not None else 0} "
                    f"n_out={len(modes_stage4)} | {t_stage4:.3f}s"
                )
        elif verbose:
            print(f"[SeedMode][Q{query_id}] Stage4 skipped")

        if verbose:
            print(f"[SeedMode][Q{query_id}] Query done | total={time.perf_counter() - t_query0:.3f}s")

        return SeedModeSearchResult(
            query_id=query_id,
            seeds_kept=seeds_kept,
            seed_clouds=seed_clouds,
            modes_init=modes_init,
            modes_before_stage3=modes_before_stage3,
            modes_final=modes_final,
            stage_trace=stage_trace,
            metadata={
                "n_seed_kept": len(seeds_kept),
                "n_modes_init": len(modes_init),
                "n_modes_before_stage3": len(modes_before_stage3),
                "n_modes_final": len(modes_final),
            },
        )

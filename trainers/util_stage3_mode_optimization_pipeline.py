from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch


@dataclass
class ModeOptimizationConfig:
    """
    Global runtime config shared by the mode optimization pipeline.

    The first implementation only fixes the protocol and common knobs.
    Concrete algorithms are expected to interpret these fields as needed.
    """

    score_mode: str = "ingp"
    metric_goal: str = "maximize"
    sampling_method: str = "sobol"
    sampling_schedule: Sequence[int] = (64, 64)
    elite_fraction: float = 0.25
    sigma_floor: float = 1e-3
    sigma_ceiling: float = 1.0
    max_shrink_rounds: int = 2
    max_modes_to_cma: int = 4
    min_active_modes: int = 1
    final_optimizer_name: str = "cma-es"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModeState:
    """
    Runtime state for one candidate mode.

    A mode represents a candidate region, not just a single point.
    The center and sigma together describe the current search distribution.
    """

    query_id: int
    mode_id: str
    center_raw: torch.Tensor
    sigma_raw: torch.Tensor
    bounds_low_raw: Optional[torch.Tensor] = None
    bounds_high_raw: Optional[torch.Tensor] = None
    score_mode: str = "ingp"
    metric_goal: str = "maximize"
    stage: str = "init"
    alive: bool = True
    budget_used: float = 0.0
    best_coord_raw: Optional[torch.Tensor] = None
    best_score: Optional[float] = None
    init_score: Optional[float] = None
    latest_metric: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def clone(self) -> "ModeState":
        return ModeState(
            query_id=self.query_id,
            mode_id=self.mode_id,
            center_raw=self.center_raw.clone(),
            sigma_raw=self.sigma_raw.clone(),
            bounds_low_raw=None if self.bounds_low_raw is None else self.bounds_low_raw.clone(),
            bounds_high_raw=None if self.bounds_high_raw is None else self.bounds_high_raw.clone(),
            score_mode=self.score_mode,
            metric_goal=self.metric_goal,
            stage=self.stage,
            alive=bool(self.alive),
            budget_used=float(self.budget_used),
            best_coord_raw=None if self.best_coord_raw is None else self.best_coord_raw.clone(),
            best_score=None if self.best_score is None else float(self.best_score),
            init_score=None if self.init_score is None else float(self.init_score),
            latest_metric=None if self.latest_metric is None else float(self.latest_metric),
            metadata=dict(self.metadata),
            history=list(self.history),
        )

    def record_event(self, stage: str, **kwargs: Any) -> None:
        event = {"stage": stage, "budget_used": float(self.budget_used)}
        event.update(kwargs)
        self.history.append(event)


@dataclass
class ModeSampleBatch:
    """
    Samples drawn inside one mode.
    """

    query_id: int
    mode_id: str
    stage: str
    coords_raw: torch.Tensor
    scores: Optional[torch.Tensor] = None
    weights: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModeEvaluation:
    """
    Scalar metric report consumed by the scheduler.

    The scheduler should rely on `metric` as the primary ranking signal.
    Additional diagnostics can be stored in `aux`.
    """

    query_id: int
    mode_id: str
    stage: str
    budget_increment: float
    total_budget: float
    metric: float
    metric_goal: str = "maximize"
    best_score: Optional[float] = None
    center_raw: Optional[torch.Tensor] = None
    sigma_raw: Optional[torch.Tensor] = None
    aux: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModeUpdateResult:
    """
    Result of one sample -> shrink cycle for a single mode.
    """

    mode_state: ModeState
    sample_batch: ModeSampleBatch
    evaluation: ModeEvaluation
    should_continue: bool = True


@dataclass
class SchedulerDecision:
    """
    Selection result for one scheduling round.
    """

    survivor_ids: List[str]
    killed_ids: List[str] = field(default_factory=list)
    promoted_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """
    Final output of the mode optimization pipeline for one query.
    """

    query_id: int
    modes_all: List[ModeState]
    surviving_modes: List[ModeState]
    final_modes: List[ModeState]
    evaluations: List[ModeEvaluation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseModeInitializer(ABC):
    """
    Build initial mode states from coarse retrieval outputs or any other context.
    """

    @abstractmethod
    def initialize_modes(
        self,
        query_id: int,
        init_context: Dict[str, Any],
        config: ModeOptimizationConfig,
    ) -> List[ModeState]:
        raise NotImplementedError


class BaseModeSampler(ABC):
    """
    Sample coordinates inside one mode.
    """

    @abstractmethod
    def sample_mode(
        self,
        mode_state: ModeState,
        n_samples: int,
        config: ModeOptimizationConfig,
    ) -> ModeSampleBatch:
        raise NotImplementedError


class BaseModeMetricEvaluator(ABC):
    """
    Evaluate sampled coordinates and return one score per sample.

    Expected output shape: [num_samples]
    """

    @abstractmethod
    def evaluate_mode_samples(
        self,
        mode_state: ModeState,
        sample_batch: ModeSampleBatch,
        query_context: Optional[Dict[str, Any]],
        config: ModeOptimizationConfig,
    ) -> torch.Tensor:
        raise NotImplementedError


class BaseModeShrinker(ABC):
    """
    Update the mode distribution from elite samples.

    Typical implementations update center and diagonal sigma, but the
    interface also allows more advanced covariance logic later.
    """

    @abstractmethod
    def update_mode(
        self,
        mode_state: ModeState,
        sample_batch: ModeSampleBatch,
        query_context: Optional[Dict[str, Any]],
        config: ModeOptimizationConfig,
    ) -> ModeUpdateResult:
        raise NotImplementedError


class BaseModeScheduler(ABC):
    """
    Decide which modes survive, are promoted, or get terminated.
    """

    @abstractmethod
    def select_modes(
        self,
        query_id: int,
        round_index: int,
        active_modes: List[ModeState],
        evaluations: List[ModeEvaluation],
        config: ModeOptimizationConfig,
    ) -> SchedulerDecision:
        raise NotImplementedError


class BaseFinalModeOptimizer(ABC):
    """
    Final high-budget optimizer, such as CMA-ES.
    """

    @abstractmethod
    def optimize_modes(
        self,
        query_id: int,
        modes: List[ModeState],
        query_context: Optional[Dict[str, Any]],
        config: ModeOptimizationConfig,
    ) -> List[ModeState]:
        raise NotImplementedError


class ModeOptimizationPipeline:
    """
    End-to-end orchestration for:

    initialize modes -> sample -> evaluate -> shrink -> schedule -> final optimize

    This skeleton keeps the stage boundaries explicit so later experiments can
    swap in ASHA, custom halving, or different mode-level optimizers.
    """

    def __init__(
        self,
        initializer: BaseModeInitializer,
        sampler: BaseModeSampler,
        metric_evaluator: BaseModeMetricEvaluator,
        shrinker: BaseModeShrinker,
        scheduler: BaseModeScheduler,
        final_optimizer: Optional[BaseFinalModeOptimizer] = None,
        config: Optional[ModeOptimizationConfig] = None,
    ):
        self.initializer = initializer
        self.sampler = sampler
        self.metric_evaluator = metric_evaluator
        self.shrinker = shrinker
        self.scheduler = scheduler
        self.final_optimizer = final_optimizer
        self.config = config or ModeOptimizationConfig()

    def run_query(
        self,
        query_id: int,
        init_context: Dict[str, Any],
        query_context: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        modes_all = self.initializer.initialize_modes(
            query_id=query_id,
            init_context=init_context,
            config=self.config,
        )
        active_modes = [mode.clone() for mode in modes_all if mode.alive]
        all_evaluations: List[ModeEvaluation] = []

        for round_index in range(int(self.config.max_shrink_rounds)):
            if len(active_modes) == 0:
                break

            n_samples = self._resolve_n_samples(round_index)
            round_updates: List[ModeUpdateResult] = []
            for mode_state in active_modes:
                sample_batch = self.sampler.sample_mode(
                    mode_state=mode_state,
                    n_samples=n_samples,
                    config=self.config,
                )
                sample_scores = self.metric_evaluator.evaluate_mode_samples(
                    mode_state=mode_state,
                    sample_batch=sample_batch,
                    query_context=query_context,
                    config=self.config,
                )
                sample_batch.scores = sample_scores
                update = self.shrinker.update_mode(
                    mode_state=mode_state,
                    sample_batch=sample_batch,
                    query_context=query_context,
                    config=self.config,
                )
                round_updates.append(update)
                all_evaluations.append(update.evaluation)

            active_modes = [update.mode_state for update in round_updates if update.should_continue and update.mode_state.alive]
            if len(active_modes) == 0:
                break

            decision = self.scheduler.select_modes(
                query_id=query_id,
                round_index=round_index,
                active_modes=active_modes,
                evaluations=[update.evaluation for update in round_updates],
                config=self.config,
            )
            active_modes = self._apply_scheduler_decision(active_modes, decision)

        surviving_modes = [mode.clone() for mode in active_modes]
        final_modes = surviving_modes
        if self.final_optimizer is not None and len(surviving_modes) > 0:
            final_modes = self.final_optimizer.optimize_modes(
                query_id=query_id,
                modes=[mode.clone() for mode in surviving_modes],
                query_context=query_context,
                config=self.config,
            )

        return PipelineResult(
            query_id=query_id,
            modes_all=modes_all,
            surviving_modes=surviving_modes,
            final_modes=final_modes,
            evaluations=all_evaluations,
            metadata={
                "sampling_schedule": tuple(self.config.sampling_schedule),
                "max_shrink_rounds": int(self.config.max_shrink_rounds),
                "final_optimizer_name": self.config.final_optimizer_name,
            },
        )

    def _resolve_n_samples(self, round_index: int) -> int:
        schedule = list(self.config.sampling_schedule)
        if len(schedule) == 0:
            raise ValueError("sampling_schedule must contain at least one positive integer.")
        idx = min(int(round_index), len(schedule) - 1)
        n_samples = int(schedule[idx])
        if n_samples <= 0:
            raise ValueError("sampling_schedule values must be > 0.")
        return n_samples

    @staticmethod
    def _apply_scheduler_decision(
        active_modes: List[ModeState],
        decision: SchedulerDecision,
    ) -> List[ModeState]:
        keep_ids = set(decision.survivor_ids)
        next_modes: List[ModeState] = []
        for mode_state in active_modes:
            mode_state.alive = mode_state.mode_id in keep_ids
            if not mode_state.alive:
                mode_state.stage = "killed"
                mode_state.record_event("killed_by_scheduler")
                continue
            mode_state.stage = "scheduled"
            mode_state.record_event("scheduled", promoted=(mode_state.mode_id in set(decision.promoted_ids)))
            next_modes.append(mode_state)
        return next_modes

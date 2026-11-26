"""Leave-one-lot-out cross-validation for choosing the mixture component count.

训练批按 ``lot_id`` 整批轮流留出：每一折用其余训练批拟合候选成分数，再对留出批
逐板评分。候选按"最差一批的平均对数密度"择优，同分取成分少的；任一折拟合失败
即淘汰该候选。

状态机本身与并发模型解耦，可在中断后按已确认的折评分恢复。回传先后不影响最终
赢家：淘汰只在数学上能断定候选必输时发生。
"""

from dataclasses import dataclass, field
from hashlib import sha256

from .mixture import evaluate
from .selection import fit_candidate

MIN_TRAINING_BOARDS = 12


class FoldError(ValueError):
    """某一折无法完成（训练板不足、候选起点未收敛等）。"""


@dataclass(frozen=True)
class FoldResult:
    key: str
    size: int
    held_out_lot: str
    train_boards: int
    held_out_boards: int
    mean_log_density: float
    converged_starts: int
    iterations: int


@dataclass(frozen=True)
class FoldRecord:
    key: str
    size: int
    held_out_lot: str
    ok: bool
    mean_log_density: float | None
    reason: str | None


@dataclass
class _CandidateState:
    size: int
    # 已确认折的平均对数密度（仅成功折）
    scores: dict[str, float] = field(default_factory=dict)
    failed_fold: str | None = None
    failed_reason: str | None = None
    pruned: bool = False
    in_flight: set[str] = field(default_factory=set)

    @property
    def failed(self) -> bool:
        return self.failed_fold is not None

    @property
    def terminal(self) -> bool:
        return self.pruned or self.failed

    def current_worst(self) -> float:
        return min(self.scores.values())


def fold_key(training_values: list[float], size: int, held_out_lot: str) -> str:
    """折的内容指纹：训练读数（已排序）、成分数与留出批相同则键相同。"""
    payload = (
        '|'.join(f'{value:.17g}' for value in training_values)
        + f'#K={size}#LOT={held_out_lot}'
    )
    return sha256(payload.encode('utf-8')).hexdigest()


def compute_fold(training_values: list[float], held_out_values: list[float],
                 size: int, held_out_lot: str) -> FoldResult:
    """执行单折：用训练读数拟合，再评留出批的逐板平均对数密度。"""
    if len(training_values) < MIN_TRAINING_BOARDS:
        raise FoldError(f'留出批 {held_out_lot} 后剩余训练板 {len(training_values)} 件，少于 {MIN_TRAINING_BOARDS} 件。')
    candidate = fit_candidate(training_values, size)
    if candidate.fit is None or not candidate.converged_starts:
        raise FoldError(f'成分数 {size} 在留出批 {held_out_lot} 的折上没有收敛起点。')
    total = sum(evaluate(candidate.fit.components, value)[0] for value in held_out_values)
    return FoldResult(
        key=fold_key(training_values, size, held_out_lot),
        size=size,
        held_out_lot=held_out_lot,
        train_boards=len(training_values),
        held_out_boards=len(held_out_values),
        mean_log_density=total / len(held_out_values),
        converged_starts=candidate.converged_starts,
        iterations=candidate.fit.iterations,
    )


class InMemoryFoldStore:
    """进程内折评分缓存，用于测试和无数据库后端。"""

    def __init__(self) -> None:
        self._records: dict[str, FoldRecord] = {}

    def load(self, keys: set[str]) -> dict[str, FoldRecord]:
        return {key: self._records[key] for key in keys if key in self._records}

    def save(self, record: FoldRecord) -> None:
        self._records.setdefault(record.key, record)

    def clear(self) -> None:
        self._records.clear()


class LotCrossValidation:
    """按 lot 轮流留出的并发交叉验证状态机。

    外部执行器通过 :meth:`plan_tasks` 取得待派发折，完成后用 :meth:`record`
    回传。回传触发必输淘汰，取消由执行器按新计划负责。状态可由持久化折缓存
    恢复，重试不会重复计分。
    """

    def __init__(self, lots: dict[str, list[float]], max_components: int,
                 store=None) -> None:
        if len(lots) < 2:
            raise FoldError('整批轮流留出至少需要两个训练批次。')
        self.lots = {lot: sorted(values) for lot, values in sorted(lots.items())}
        self.lot_order = tuple(self.lots)
        self.sizes = tuple(range(1, max_components + 1))
        self.store = store if store is not None else InMemoryFoldStore()
        self.states = {size: _CandidateState(size) for size in self.sizes}
        self._restore()

    # ---- 折的输入与缓存 -------------------------------------------------

    def _fold_inputs(self, size: int, held_out_lot: str):
        training_values = [
            value for lot, values in self.lots.items()
            if lot != held_out_lot for value in values
        ]
        return training_values, self.lots[held_out_lot]

    def _all_keys(self) -> dict[tuple[int, str], str]:
        keys = {}
        for size in self.sizes:
            for lot in self.lot_order:
                training_values, _ = self._fold_inputs(size, lot)
                keys[(size, lot)] = fold_key(training_values, size, lot)
        return keys

    def _restore(self) -> None:
        """从缓存载入已确认折，沿用评分且不重复计分。"""
        keys = set(self._all_keys().values())
        for record in self.store.load(keys).values():
            state = self.states[record.size]
            if record.ok:
                state.scores.setdefault(record.held_out_lot, record.mean_log_density)
            elif state.failed_fold is None:
                state.failed_fold = record.held_out_lot
                state.failed_reason = record.reason
        self._apply_pruning()

    # ---- 派发计划 -------------------------------------------------------

    def plan_tasks(self) -> list[tuple[int, str, str]]:
        """返回 ``(成分数, 留出批, 折键)`` 列表，已确认/在途/终止候选不再派发。"""
        tasks = []
        for size in self.sizes:
            state = self.states[size]
            if state.terminal:
                continue
            for lot in self.lot_order:
                if lot in state.scores or lot == state.failed_fold or lot in state.in_flight:
                    continue
                training_values, _ = self._fold_inputs(size, lot)
                tasks.append((size, lot, fold_key(training_values, size, lot)))
        return tasks

    def mark_dispatched(self, size: int, lot: str) -> None:
        state = self.states[size]
        if not state.terminal and lot not in state.scores and lot != state.failed_fold:
            state.in_flight.add(lot)

    # ---- 回传与淘汰 -----------------------------------------------------

    def record(self, result: FoldResult | None, size: int, held_out_lot: str,
               error: str | None = None) -> None:
        """幂等回传：同一折重复回传不重复计分，回传先后不影响赢家。"""
        state = self.states[size]
        state.in_flight.discard(held_out_lot)
        # 候选已终止（失败或被证明必输）时，迟到/重复回传一律丢弃
        if state.terminal or held_out_lot in state.scores or state.failed_fold == held_out_lot:
            self._apply_pruning()
            return
        training_values, held_out_values = self._fold_inputs(size, held_out_lot)
        key = fold_key(training_values, size, held_out_lot)
        if held_out_lot in state.scores or state.failed_fold == held_out_lot:
            self._apply_pruning()
            return
        if error is not None or result is None:
            state.failed_fold = held_out_lot
            state.failed_reason = error or '折拟合失败。'
            self.store.save(FoldRecord(key, size, held_out_lot, False, None, state.failed_reason))
        else:
            state.scores[held_out_lot] = result.mean_log_density
            self.store.save(FoldRecord(key, size, held_out_lot, True,
                                       result.mean_log_density, None))
        self._apply_pruning()

    def _apply_pruning(self) -> None:
        """必输才淘汰：已有更小的完成候选，其最差折不低于本候选当前最差折。"""
        for size in self.sizes:
            state = self.states[size]
            if state.terminal or not state.scores:
                continue
            worst = state.current_worst()
            for smaller in (candidate for candidate in self.sizes if candidate < size):
                rival = self.states[smaller]
                if rival.terminal or len(rival.scores) < len(self.lot_order):
                    continue
                # 同分较小候选必胜；rival 最差折更高也必胜
                if rival.current_worst() >= worst:
                    state.pruned = True
                    state.in_flight.clear()
                    break

    # ---- 结果 -----------------------------------------------------------

    @property
    def done(self) -> bool:
        return not self.plan_tasks() and not any(state.in_flight for state in self.states.values())

    def _completed(self) -> list[_CandidateState]:
        total = len(self.lot_order)
        return [state for state in self.states.values()
                if not state.terminal and len(state.scores) == total]

    @property
    def winner(self) -> _CandidateState | None:
        completed = self._completed()
        if not completed:
            return None
        return min(completed, key=lambda state: (-state.current_worst(), state.size))

    def diagnostics(self) -> list[dict]:
        rows = []
        total = len(self.lot_order)
        for size in self.sizes:
            state = self.states[size]
            if state.failed:
                status = 'fold_failed'
            elif state.pruned:
                status = 'pruned_loser'
            elif len(state.scores) == total:
                status = 'eligible'
            else:
                status = 'pending'
            rows.append({
                'size': size,
                'status': status,
                'scored_folds': len(state.scores),
                'total_folds': total,
                'worst_mean_log_density': state.current_worst() if state.scores else None,
                'failed_fold': state.failed_fold,
                'failed_reason': state.failed_reason,
            })
        return rows

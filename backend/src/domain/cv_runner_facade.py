"""领域层并发门面：线程池内只做纯计算，存储与框架留在服务层。

调度采用有界在途窗口：每次有折回传就重新计算派发计划，已被证明必输的候选
其剩余折不再派发，已在途的折回传后丢弃结果、不计分。
"""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

from . import cross_validation
from .cross_validation import FoldError, FoldResult, LotCrossValidation


@dataclass
class CrossValidationRun:
    cv: LotCrossValidation
    results: dict[tuple[int, str], FoldResult] = field(default_factory=dict)
    dispatched: int = 0
    skipped_by_pruning: int = 0

    @property
    def winner_size(self) -> int | None:
        winner = self.cv.winner
        return winner.size if winner is not None else None


def _run_fold(cv: LotCrossValidation, size: int, lot: str):
    training, held = cv._fold_inputs(size, lot)
    try:
        # 经模块属性调用，测试可替换 cross_validation.compute_fold 注入慢折
        return size, lot, cross_validation.compute_fold(training, held, size, lot), None
    except FoldError as error:
        return size, lot, None, str(error)
    except Exception as error:  # 起点全未收敛等数值失败统一按折失败处理
        return size, lot, None, f'折拟合失败：{error}'


def run_cross_validation(cv: LotCrossValidation, max_workers: int = 4) -> CrossValidationRun:
    """增量并发派发折任务，直到全部候选终止且在途折全部回收。"""
    outcome = CrossValidationRun(cv=cv)
    in_flight: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while True:
            # 回传后重算计划：必输候选的剩余折不会再出现在计划中。
            wanted = {(size, lot) for size, lot, _key in cv.plan_tasks()}
            for size, lot in sorted(wanted):
                if len(in_flight) >= max_workers:
                    break
                if (size, lot) in in_flight:
                    continue
                cv.mark_dispatched(size, lot)
                outcome.dispatched += 1
                in_flight[(size, lot)] = pool.submit(_run_fold, cv, size, lot)
            if not in_flight:
                break
            done, _ = wait(in_flight.values(), return_when=FIRST_COMPLETED)
            for future in done:
                size, lot = next(key for key, value in in_flight.items() if value is future)
                del in_flight[(size, lot)]
                result_size, result_lot, result, error = future.result()
                state = cv.states[result_size]
                if state.terminal:
                    # 回传期间候选已被证明必输（或已失败），结果丢弃且不计分
                    outcome.skipped_by_pruning += 1
                    state.in_flight.discard(result_lot)
                    continue
                cv.record(result, result_size, result_lot, error)
                if result is not None:
                    outcome.results[(result_size, result_lot)] = result
    return outcome

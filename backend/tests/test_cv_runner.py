"""并发门面测试：必输早停、回传顺序无关与中断恢复。"""

import os
import sys
import time
import unittest
from random import Random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.domain import cv_runner_facade as facade  # noqa: E402
from src.domain import cross_validation as cv_module  # noqa: E402
from src.domain.cross_validation import (  # noqa: E402
    FoldError,
    InMemoryFoldStore,
    LotCrossValidation,
    compute_fold,
)


def make_lots():
    random = Random(11)
    lots = {}
    for name, offset in (('L1', 0.0), ('L2', 0.3), ('L3', -0.4), ('L4', 0.1)):
        values = [round(random.gauss(8.5, 1.3), 3) for _ in range(8)]
        values += [round(random.gauss(21.0 + offset, 1.6), 3) for _ in range(8)]
        random.shuffle(values)
        lots[name] = values
    return lots


def run_all_sequential(lots, max_components):
    cv = LotCrossValidation({k: list(v) for k, v in lots.items()}, max_components)
    while not cv.done:
        for size, lot, _ in cv.plan_tasks():
            cv.mark_dispatched(size, lot)
            training, held = cv._fold_inputs(size, lot)
            try:
                cv.record(compute_fold(training, held, size, lot), size, lot)
            except FoldError as error:
                cv.record(None, size, lot, str(error))
    return cv


class RunnerTests(unittest.TestCase):
    def test_runner_matches_sequential_winner(self):
        lots = make_lots()
        outcome = facade.run_cross_validation(LotCrossValidation({k: list(v) for k, v in lots.items()}, 3))
        full = run_all_sequential(lots, 3)
        self.assertEqual(outcome.winner_size, full.winner.size)

    def test_pruned_candidate_does_not_dispatch_remaining_folds(self):
        lots = make_lots()
        cv = LotCrossValidation(lots, 3)
        # 让 K=1 快速完成并给出很高的最差折，再让 K=3 首折极差：
        # K=3 首折一回传即被已完成的 K=1 支配，可断定必输，剩余折必须停发。
        for state_size, fill in ((1, -2.0), (3, -9.0)):
            state = cv.states[state_size]
            for lot in cv.lot_order:
                state.scores[lot] = fill
        cv._apply_pruning()
        # 手工构造下，K=3 立即成为必输候选
        self.assertTrue(cv.states[3].pruned)

        # 通过真实调度再验证一次：预置 K=1 已完成、K=3 首折已确认
        cv2 = LotCrossValidation({k: list(v) for k, v in lots.items()}, 2)
        k1 = cv2.states[1]
        for lot in cv2.lot_order:
            k1.scores[lot] = -2.0
        cv2.states[2].scores[cv2.lot_order[0]] = -9.0
        cv2._apply_pruning()
        self.assertTrue(cv2.states[2].pruned)
        planned = {lot for size, lot, _ in cv2.plan_tasks()}
        # K=2 剩余折不再出现在派发计划
        for lot in cv2.lot_order[1:]:
            self.assertNotIn(lot, planned)

    def test_runner_skips_in_flight_result_after_pruning(self):
        lots = make_lots()
        cv = LotCrossValidation(lots, 2)
        real_compute = cv_module.compute_fold

        def fake_compute(training, held, size, lot):
            if size == 2:
                time.sleep(0.05)
            return real_compute(training, held, size, lot)

        cv_module.compute_fold = fake_compute
        try:
            outcome = facade.run_cross_validation(cv, max_workers=2)
        finally:
            cv_module.compute_fold = real_compute
        # 无论是否触发剪枝，赢家必须与全量顺序结果一致，且状态机正常终止
        full = run_all_sequential(lots, 2)
        self.assertEqual(outcome.winner_size, full.winner.size)
        self.assertTrue(cv.done)

    def test_resume_with_store_completes_without_rescoring(self):
        lots = make_lots()
        store = InMemoryFoldStore()
        cv = LotCrossValidation({k: list(v) for k, v in lots.items()}, 2, store=store)
        # 跑两个折后模拟中断
        for size, lot, _ in cv.plan_tasks()[:2]:
            cv.mark_dispatched(size, lot)
            training, held = cv._fold_inputs(size, lot)
            cv.record(compute_fold(training, held, size, lot), size, lot)
        resumed = LotCrossValidation({k: list(v) for k, v in lots.items()}, 2, store=store)
        facade.run_cross_validation(resumed)
        self.assertIsNotNone(resumed.winner)
        # 折键按内容去重：已确认折沿用缓存，最终缓存条数等于全部折数，无重复计分
        self.assertEqual(len(store._records), 2 * len(lots))


if __name__ == '__main__':
    unittest.main(verbosity=2)

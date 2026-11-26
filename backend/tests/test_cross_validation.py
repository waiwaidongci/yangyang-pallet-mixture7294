"""整批轮流留出交叉验证的纯领域测试（仅依赖标准库）。"""

import os
import sys
import unittest
from random import Random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.domain.cross_validation import (  # noqa: E402
    FoldError,
    InMemoryFoldStore,
    LotCrossValidation,
    compute_fold,
    fold_key,
)


def make_lots(seed, spec):
    random = Random(seed)
    lots = {}
    for lot, n_dry, n_wet, offset in spec:
        values = [round(random.gauss(8.5 + offset, 1.4), 3) for _ in range(n_dry)]
        values += [round(random.gauss(21.0 + offset, 1.7), 3) for _ in range(n_wet)]
        random.shuffle(values)
        lots[lot] = values
    return lots


class CrossValidationTests(unittest.TestCase):
    def setUp(self):
        # 三个训练批，每批 16 件，保证任一折后仍 >=12 件
        self.lots = make_lots(7, [
            ('LOT-A', 8, 8, 0.0),
            ('LOT-B', 8, 8, 0.4),
            ('LOT-C', 8, 8, -0.5),
        ])

    def _run_to_completion(self, cv, order=None):
        """模拟并发执行器：按给定回传顺序跑完所有折，回传结果幂等。"""
        dispatched = set()
        while not cv.done:
            tasks = cv.plan_tasks()
            self.assertTrue(tasks)
            if order:
                keyed = {(size, lot): task for size, lot, task in tasks}
                pending = [item for item in order if item in keyed and item not in dispatched]
                chosen = [pending[0]] if pending else [tasks[0][:2]]
            else:
                chosen = [tasks[0][:2]]
            for size, lot in chosen:
                dispatched.add((size, lot))
                cv.mark_dispatched(size, lot)
                training, held = cv._fold_inputs(size, lot)
                try:
                    result = compute_fold(training, held, size, lot)
                    cv.record(result, size, lot)
                except FoldError as error:
                    cv.record(None, size, lot, str(error))

    def test_deterministic_fold_keys_and_completes(self):
        cv = LotCrossValidation(self.lots, 3)
        self.assertEqual(len(cv.lot_order), 3)
        self._run_to_completion(cv)
        winner = cv.winner
        self.assertIsNotNone(winner)
        # 两个成分应在双群数据上胜出单成分
        self.assertEqual(winner.size, 2)
        for state in cv.states.values():
            self.assertEqual(len(state.scores), 3)

    def test_winner_uses_worst_fold_and_callback_order_independent(self):
        # 固定数据下，不同回传顺序必须得到同一赢家
        winners = []
        orders = [
            None,
            [(1, 'LOT-A'), (2, 'LOT-A'), (3, 'LOT-A')],
            [(3, 'LOT-C'), (3, 'LOT-B'), (3, 'LOT-A')],
        ]
        for order in orders:
            cv = LotCrossValidation({k: list(v) for k, v in self.lots.items()}, 3)
            self._run_to_completion(cv, order)
            winners.append(cv.winner.size)
        self.assertEqual(len(set(winners)), 1)

        # 赢家的判定值确实是各候选最差折的最大值
        completed = [s for s in cv.states.values() if len(s.scores) == 3]
        worsts = {s.size: min(s.scores.values()) for s in completed}
        best_size = min(worsts, key=lambda k: (-worsts[k], k))
        self.assertEqual(cv.winner.size, best_size)

    def test_duplicate_report_is_idempotent(self):
        cv = LotCrossValidation(self.lots, 2)
        tasks = cv.plan_tasks()
        size, lot, _ = tasks[0]
        cv.mark_dispatched(size, lot)
        training, held = cv._fold_inputs(size, lot)
        result = compute_fold(training, held, size, lot)
        cv.record(result, size, lot)
        first = dict(cv.states[size].scores)
        # 重复回传同一折（重试场景）不重复计分
        cv.record(result, size, lot)
        cv.record(result, size, lot)
        self.assertEqual(cv.states[size].scores, first)

    def test_resume_reuses_confirmed_scores(self):
        store = InMemoryFoldStore()
        cv = LotCrossValidation(self.lots, 2, store=store)
        # 只跑前两个派发的折就"中断"
        tasks = cv.plan_tasks()[:2]
        for size, lot, _ in tasks:
            cv.mark_dispatched(size, lot)
            training, held = cv._fold_inputs(size, lot)
            cv.record(compute_fold(training, held, size, lot), size, lot)
        cached = len(store._records)
        self.assertGreater(cached, 0)

        # 新实例从同一缓存恢复，已确认折不再出现在计划中
        resumed = LotCrossValidation({k: list(v) for k, v in self.lots.items()}, 2, store=store)
        planned = {(size, lot) for size, lot, _ in resumed.plan_tasks()}
        for size, lot, _ in tasks:
            self.assertNotIn((size, lot), planned)
            self.assertIn(lot, resumed.states[size].scores)
        self._run_to_completion(resumed)
        self.assertIsNotNone(resumed.winner)
        # 全部折数 = 成分数 × 批次数；缓存按折键去重，已确认折没有重复计分
        total_folds = len(cv.sizes) * len(cv.lot_order)
        self.assertEqual(len(store._records), total_folds)

    def test_failed_fold_eliminates_candidate(self):
        # 制造一个折后训练板不足的场景：两批，其中一批只有 1 件
        tiny = {'BIG': self.lots['LOT-A'] + self.lots['LOT-B'], 'TINY': [9.1]}
        cv = LotCrossValidation(tiny, 2)
        # 留出 BIG 后只剩 1 件，所有成分数都应失败
        for size in cv.sizes:
            training, held = cv._fold_inputs(size, 'BIG')
            with self.assertRaises(FoldError):
                compute_fold(training, held, size, 'BIG')
            cv.record(None, size, 'BIG', '训练板不足。')
        self.assertTrue(all(state.failed for state in cv.states.values()))
        self.assertIsNone(cv.winner)
        self.assertTrue(cv.done)

    def test_no_qualified_candidate_reports_failure(self):
        cv = LotCrossValidation(self.lots, 3)
        # 人为令每个候选都有一折失败
        for size in cv.sizes:
            cv.record(None, size, 'LOT-A', '模拟失败。')
        self.assertTrue(cv.done)
        self.assertIsNone(cv.winner)
        self.assertTrue(all(row['status'] == 'fold_failed' for row in cv.diagnostics()))

    def test_pruning_only_removes_proven_losers(self):
        cv = LotCrossValidation(self.lots, 3)
        # 先把 K=1 全部跑完
        for lot in cv.lot_order:
            cv.mark_dispatched(1, lot)
            training, held = cv._fold_inputs(1, lot)
            cv.record(compute_fold(training, held, 1, lot), 1, lot)
        # K=1 完成；K=2/K=3 各跑一折后，若当前最差折已被 K=1 支配则必输淘汰
        for size in (2, 3):
            lot = cv.lot_order[0]
            cv.mark_dispatched(size, lot)
            training, held = cv._fold_inputs(size, lot)
            cv.record(compute_fold(training, held, size, lot), size, lot)
        # 与"全量不剪枝"跑一遍对照：赢家必须一致
        full = LotCrossValidation({k: list(v) for k, v in self.lots.items()}, 3)
        self._run_to_completion(full)
        # 被剪枝候选不会再派发折
        planned = {(size, lot) for size, lot, _ in cv.plan_tasks()}
        for size, state in cv.states.items():
            if state.pruned:
                self.assertTrue(all((size, lot) not in planned for lot in cv.lot_order))
        # 继续跑完未淘汰候选，赢家与全量一致
        self._run_to_completion(cv)
        self.assertEqual(cv.winner.size, full.winner.size)

    def test_tie_prefers_smaller_component(self):
        # 手工构造状态：K=1、K=2 完成且最差折完全相同
        cv = LotCrossValidation(self.lots, 2)
        for state in cv.states.values():
            for lot in cv.lot_order:
                state.scores[lot] = -3.0
        winner = cv.winner
        self.assertEqual(winner.size, 1)

    def test_single_training_lot_rejected(self):
        with self.assertRaises(FoldError):
            LotCrossValidation({'ONLY': [1.0] * 14}, 2)

    def test_fold_key_is_content_based(self):
        values = sorted(self.lots['LOT-A'] + self.lots['LOT-B'])
        key1 = fold_key(values, 2, 'LOT-C')
        key2 = fold_key(list(values), 2, 'LOT-C')
        key3 = fold_key(values, 3, 'LOT-C')
        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)


if __name__ == '__main__':
    unittest.main(verbosity=2)

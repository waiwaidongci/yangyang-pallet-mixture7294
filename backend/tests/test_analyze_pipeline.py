"""analyze 端到端测试：演示台账、折缓存复用与失败上报（纯标准库）。"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.domain.cross_validation import InMemoryFoldStore  # noqa: E402
from src.domain.fixtures import demo_rows  # noqa: E402
from src.domain.project import analyze  # noqa: E402


def table_by_key(result, key):
    return next(table for table in result['tables'] if table['key'] == key)


class AnalyzeEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.rows = demo_rows()

    def test_demo_ledger_selects_two_components_and_scores_all_folds(self):
        store = InMemoryFoldStore()
        result = analyze(self.rows, {}, fold_store=store)
        self.assertEqual(result['metrics']['选中成分数'], 2)
        cv_summary = table_by_key(result, 'cv_candidates')
        statuses = {row['components']: row['status'] for row in cv_summary['rows']}
        self.assertEqual(statuses[2], 'eligible')
        # 三个训练批 × 三个候选 = 9 折，首轮全部派发；折表保留每个折的去向
        folds = table_by_key(result, 'cv_fold_scoring')
        self.assertEqual(len(folds['rows']), 9)
        self.assertLessEqual(len(store._records), 9)
        # 选中候选的三折都已评分
        chosen = [row for row in folds['rows'] if row['components'] == 2]
        self.assertEqual({row['status'] for row in chosen}, {'scored'})
        self.assertEqual(len([row for row in chosen if row['status'] == 'scored']), 3)
        # BIC 表仍保留为诊断
        bic = table_by_key(result, 'bic_candidates')
        self.assertTrue(all(row['bic'] is not None for row in bic['rows']))
        # 冻结模型按均值升序，干组在前湿组在后
        components = table_by_key(result, 'model_components')
        means = [row['mean_pct'] for row in components['rows']]
        self.assertEqual(means, sorted(means))

    def test_second_analyze_reuses_confirmed_fold_scores(self):
        store = InMemoryFoldStore()
        first = analyze(self.rows, {}, fold_store=store)
        cached_keys = set(store._records)
        second = analyze(self.rows, {}, fold_store=store)
        # 同一台账重跑：折键完全一致，没有新增或覆盖任何已确认评分
        self.assertEqual(set(store._records), cached_keys)
        self.assertEqual(second['metrics']['选中成分数'], first['metrics']['选中成分数'])
        self.assertEqual(second['metrics']['最差留出批平均对数密度'],
                         first['metrics']['最差留出批平均对数密度'])

    def test_single_training_lot_is_failure(self):
        rows = [dict(row, lot_id='LONE') for row in self.rows if row['split'] == 'train']
        with self.assertRaisesRegex(ValueError, '至少需要两个训练批次'):
            analyze(rows, {})

    def test_no_qualified_candidate_reports_failure(self):
        # 两个训练批，其中一批只有 1 件：留出大批后剩余训练板不足，全部折失败
        rows = [dict(row) for row in self.rows]
        big_lot = next(iter({row['lot_id'] for row in rows if row['split'] == 'train'}))
        for row in rows:
            if row['split'] == 'train' and row['lot_id'] != big_lot:
                row['lot_id'] = big_lot
        rows.append({'id': 'TINY-001', 'lot_id': 'LOT-TINY', 'location': 'RACK-A',
                     'split': 'train', 'moisture_pct': 9.0})
        with self.assertRaisesRegex(ValueError, '没有合格候选'):
            analyze(rows, {})


if __name__ == '__main__':
    unittest.main(verbosity=2)

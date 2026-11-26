"""Domain integration for the pallet-moisture ledger and background worker."""

from collections import defaultdict

from .cross_validation import InMemoryFoldStore, LotCrossValidation
from .fixtures import demo_rows
from .records import normalize_rows, partition
from .reports import (
    board_table,
    candidate_table,
    component_table,
    cv_candidate_table,
    cv_fold_table,
    location_table,
)
from .selection import fit_candidate, select_model
from .settings import SPEC, parse_settings
from .sorting import location_diagnostics, score_boards

__all__ = ['SPEC', 'demo_rows', 'normalize_rows', 'analyze']


def analyze(rows: list[dict], settings: dict, fold_store=None) -> dict:
    options = parse_settings(settings)
    train, holdout, pending = partition(rows)
    values = [row['moisture_pct'] for row in train]

    # 整批轮流留出：训练批按 lot_id 分组，每折留出一整批。
    lots = defaultdict(list)
    for row in train:
        lots[row['lot_id']].append(row['moisture_pct'])
    if len(lots) < 2:
        raise ValueError('整批轮流留出至少需要两个训练批次，当前训练台账只有一个批次。')

    store = fold_store if fold_store is not None else InMemoryFoldStore()
    cv = LotCrossValidation(dict(lots), options['max_components'], store=store)
    # 延迟导入并发执行器，保持领域模块不依赖服务层。
    from .cv_runner_facade import run_cross_validation

    outcome = run_cross_validation(cv)
    winner_size = outcome.winner_size
    if winner_size is None:
        raise ValueError('整批轮流留出交叉验证没有合格候选：每个候选成分数都有折拟合失败。')

    # 胜出后用全部训练批重新拟合，作为冻结模型。
    refit = fit_candidate(values, winner_size)
    if refit.fit is None or not refit.converged_starts:
        raise ValueError(f'胜出候选（{winner_size} 个成分）在全量训练批上未能收敛。')
    _, bic_candidates = select_model(values, options['max_components'])
    selected = next(item for item in bic_candidates if item.size == winner_size)
    components = refit.fit.components

    trained_locations = {row['location'] for row in train}
    train_scores = score_boards(train, components, options, trained_locations)
    holdout_scores = score_boards(holdout, components, options, trained_locations)
    queue = score_boards(pending, components, options, trained_locations)
    manual_count = sum(row['action'] == 'retest' for row in queue)
    holdout_mean = sum(row['log_density'] for row in holdout_scores) / len(holdout) if holdout else None
    winner_state = cv.states[winner_size]
    warnings = []
    if selected.size == 1:
        warnings.append('整批留出选择单成分，无法确定相对干湿组，待分拣板件全部转人工复测。')
    if not holdout:
        warnings.append('台账未提供独立留出批次，缺少跨批密度诊断。')
    if not pending:
        warnings.append('台账未提供待分拣板件。')
    if any(not row['location_seen_in_train'] for row in holdout_scores + queue):
        warnings.append('留出或待分拣台账包含训练未覆盖的库位，待分拣板件须人工复测。')
    if any(item['status'] == 'fold_failed' for item in cv.diagnostics()):
        warnings.append('有候选成分数因某折拟合失败被淘汰，可在交叉验证表查看原因。')
    if any(item['status'] == 'pruned_loser' for item in cv.diagnostics()):
        warnings.append('有候选在能断定必输后提前终止派发，未跑满全部留出折。')
    return {
        'summary': (f'使用 {len(train)} 件训练板材、{len(lots)} 个训练批做整批轮流留出，'
                    f'按最差一批平均对数密度选择 {selected.size} 个成分（最差折 {winner_state.current_worst():.4f}），'
                    f'并用全部训练批重新拟合冻结模型。留出密度诊断覆盖 {len(holdout)} 件，'
                    f'{len(queue)} 件待分拣板材中 {manual_count} 件需人工复测。'),
        'metrics': {'训练板件数': len(train), '训练批次数': len(lots), '留出板件数': len(holdout),
                    '待分拣板件数': len(queue), '选中成分数': selected.size,
                    '选型方式': '整批轮流留出交叉验证（最差一批平均对数密度，同分取成分少）',
                    '选中 BIC': round(selected.bic, 10) if selected.bic is not None else None,
                    '最差留出批平均对数密度': round(winner_state.current_worst(), 10),
                    '交叉验证派发折数': outcome.dispatched,
                    '必输早停跳过折数': outcome.skipped_by_pruning,
                    '留出平均对数密度': round(holdout_mean, 10) if holdout_mean is not None else None,
                    '人工复测板件数': manual_count},
        'tables': [board_table('sorting_queue', '待分拣板件与人工复测单', queue),
                   cv_candidate_table(cv, winner_size),
                   cv_fold_table(outcome, cv),
                   candidate_table(bic_candidates, selected, train),
                   component_table(refit, train, options),
                   board_table('holdout_scoring', '整批留出板件评分', holdout_scores),
                   location_table(location_diagnostics(train_scores + holdout_scores + queue))],
        'warnings': warnings,
    }

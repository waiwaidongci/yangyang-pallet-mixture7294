"""折评分的数据库缓存：已确认折按内容指纹持久化，支持中断续跑。"""

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.config.database import Session
from src.domain.cross_validation import FoldRecord
from src.models.cv_fold import CvFoldScore
from src.utils.time import utc_stamp


class DatabaseFoldStore:
    """与 :class:`InMemoryFoldStore` 同形的数据库适配器。"""

    def load(self, keys: set[str]) -> dict[str, FoldRecord]:
        if not keys:
            return {}
        with Session() as session:
            rows = session.scalars(select(CvFoldScore).where(CvFoldScore.fold_key.in_(keys))).all()
            return {
                row.fold_key: FoldRecord(
                    key=row.fold_key,
                    size=row.components,
                    held_out_lot=row.held_out_lot,
                    ok=row.ok,
                    mean_log_density=row.mean_log_density,
                    reason=row.reason,
                )
                for row in rows
            }

    def save(self, record: FoldRecord) -> None:
        with Session.begin() as session:
            # 主键冲突说明该折已确认（并发或重试），沿用既有评分。
            session.add(CvFoldScore(
                fold_key=record.key,
                components=record.size,
                held_out_lot=record.held_out_lot,
                ok=record.ok,
                mean_log_density=record.mean_log_density,
                reason=record.reason,
                created_at=utc_stamp(),
            ))
            try:
                session.flush()
            except IntegrityError:
                session.rollback()

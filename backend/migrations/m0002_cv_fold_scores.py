"""幂等补充交叉验证折评分表（create_all 已覆盖新库，这里兼容旧库）。"""

from sqlalchemy import inspect, text

from src.config.database import engine
from src.models.cv_fold import CvFoldScore

_DDL = text(
    """
    CREATE TABLE IF NOT EXISTS cv_fold_scores (
        fold_key VARCHAR(64) PRIMARY KEY,
        components INTEGER NOT NULL,
        held_out_lot VARCHAR(40) NOT NULL,
        ok BOOLEAN NOT NULL,
        mean_log_density FLOAT NULL,
        reason TEXT NULL,
        created_at VARCHAR(32) NOT NULL
    )
    """
)


def upgrade():
    inspector = inspect(engine)
    if CvFoldScore.__tablename__ not in inspector.get_table_names():
        with engine.begin() as connection:
            connection.execute(_DDL)
            connection.execute(text('CREATE INDEX IF NOT EXISTS ix_cv_fold_scores_components '
                                    'ON cv_fold_scores (components)'))


if __name__ == '__main__':
    upgrade()

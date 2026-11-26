from sqlalchemy import Boolean, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.config.database import Base


class CvFoldScore(Base):
    """整批留出交叉验证中已确认的单折评分。

    ``fold_key`` 是折内容指纹（训练读数、成分数、留出批），作为主键保证中断
    续跑沿用已确认评分、重试不重复计分。
    """

    __tablename__ = 'cv_fold_scores'

    fold_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    components: Mapped[int] = mapped_column(Integer, index=True)
    held_out_lot: Mapped[str] = mapped_column(String(40))
    ok: Mapped[bool] = mapped_column(Boolean)
    mean_log_density: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String(32))

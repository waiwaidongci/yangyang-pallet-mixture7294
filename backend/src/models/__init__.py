"""Register all SQLAlchemy tables in every entry point using this package."""

from .account import Account
from .audit import Audit
from .cv_fold import CvFoldScore
from .dataset import Dataset
from .review import Review
from .run import Run

__all__ = ['Account', 'Audit', 'CvFoldScore', 'Dataset', 'Review', 'Run']

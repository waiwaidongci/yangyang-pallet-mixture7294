# 数据库初始化

`bootstrap.py` 按 SQLAlchemy 模型初始化表结构。后端启动时执行，worker 在后端健康后启动。演示台账与账号使用自然键补齐，后续更新数据时通过导入建立独立快照。

`m0002_cv_fold_scores.py` 幂等补充整批留出交叉验证的折评分表 `cv_fold_scores`，新库由 `create_all` 直接建表，旧库启动时按 `IF NOT EXISTS` 补齐。

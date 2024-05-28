import evalml
from evalml import AutoMLSearch
from evalml.utils import infer_feature_types


# ————————————————分类问题————————————————————————

automl = AutoMLSearch(
    X_train=X,
    y_train=y,
    problem_type="binary",
    objective="accuracy"
)

automl.search()

## 显示结果
best_pipeline = automl.best_pipeline
print(best_pipeline)










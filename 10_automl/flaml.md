tutorial： https://microsoft.github.io/FLAML/docs/Examples/AutoML-Regression

代码文件： https://microsoft.github.io/FLAML/docs/reference/automl/automl/

# —————————————————回归模型—————————————————————————————

from flaml import AutoML

初始化 AutoML 对象
automl = AutoML()

设置 AutoML 参数
automl_settings = {
    "time_budget": 3600,  # 1小时
    "metric": "r2",
    "task": "regression",
    "log_file_name": "california.log",
    "estimator_list": ["lgbm", "xgboost", "rf", "catboost"],  # 使用的模型类型
    "log_type": "all",  # 日志类型
    "model_history": True,  # 是否保存模型训练历史
    "ensemble": True,  # 是否启用集成
    "n_jobs": -1,  # 使用所有可用的CPU
    "eval_method": "cv",  # 交叉验证
    "split_type": "uniform",  # 数据分割方式
    "n_splits": 5,  # 交叉验证折数
    "mem_thres": 1024 * 1024 * 1024 * 10,  # 内存限制 (10GB)
    "log_metrics": ["r2", "mse", "mae"],  # 要记录的额外性能指标
    "train_time_limit": 3600,  # 训练时间限制
    "use_ray": False,  # 是否使用Ray进行分布式训练
    "use_ray_tune": False,  # 是否使用Ray Tune进行超参数优化
    "early_stop": True,  # 是否启用早停
    "early_stop_patience": 10,  # 早停耐心次数
    "starting_points": "auto",  # 自动设置初始搜索点
    "use_gpu": False,  # 是否使用GPU
    "max_iter": 1000,  # 最大迭代次数
    "ensemble_strategy": "bagging",  # 集成策略
    "ensemble_size": 20,  # 集成模型数量
    "save_model": True,  # 是否保存最佳模型
    "verbosity": 2,  # 日志级别
}

automl_settings = {
    "time_budget": 300,  # 1小时
    "metric": "r2",
    "task": "regression",
    "log_file_name": "flaml01.log",
    "estimator_list": ["lgbm", "xgboost", "rf", "catboost"], 
    "model_history": True, 
    "ensemble": True, 
    "n_jobs": -1, 
    "eval_method": "cv", 
    "split_type": "uniform", 
    "n_splits": 5, 
}


运行 AutoML
automl.fit(X_train=X_train, y_train=y_train, **automl_settings)

输出结果
print("Best model:", automl.best_estimator)
print("Best hyperparameters:", automl.best_config)
print("Best R2 score:", automl.best_loss)

保存最佳模型
import joblib
joblib.dump(automl.model, 'best_model.pkl')

加载最佳模型
best_model = joblib.load('best_model.pkl')



# ——————————————————————分类模型——————————————————————————————————

from flaml import AutoML
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

加载示例数据集
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

初始化 AutoML 对象
automl = AutoML()

设置 AutoML 参数
automl_settings = {
    "time_budget": 3600,  # 1小时
    "metric": "accuracy",
    "task": "classification",
    "log_file_name": "classification.log",
    "estimator_list": ["lgbm", "xgboost", "rf", "catboost"],  # 使用的模型类型
    "log_type": "all",  # 日志类型
    "model_history": True,  # 是否保存模型训练历史
    "ensemble": True,  # 是否启用集成
    "n_jobs": -1,  # 使用所有可用的CPU
    "eval_method": "cv",  # 交叉验证
    "split_type": "uniform",  # 数据分割方式
    "n_splits": 5,  # 交叉验证折数
    "mem_thres": 1024 * 1024 * 1024 * 10,  # 内存限制 (10GB)
    "log_metrics": ["accuracy", "f1", "precision", "recall"],  # 要记录的额外性能指标
    "train_time_limit": 3600,  # 训练时间限制
    "use_ray": False,  # 是否使用Ray进行分布式训练
    "use_ray_tune": False,  # 是否使用Ray Tune进行超参数优化
    "early_stop": True,  # 是否启用早停
    "early_stop_patience": 10,  # 早停耐心次数
    "starting_points": "auto",  # 自动设置初始搜索点
    "use_gpu": False,  # 是否使用GPU
    "max_iter": 1000,  # 最大迭代次数
    "ensemble_strategy": "bagging",  # 集成策略
    "ensemble_size": 20,  # 集成模型数量
    "save_model": True,  # 是否保存最佳模型
    "verbosity": 2,  # 日志级别
    "custom_hp": {  # 自定义超参数搜索空间
        "lgbm": {
            "num_leaves": {"domain": "loguniform", "low": 1, "high": 100},
            "learning_rate": {"domain": "uniform", "low": 1e-4, "high": 1e-1},
        },
        "xgboost": {
            "max_depth": {"domain": "uniform", "low": 3, "high": 10},
            "learning_rate": {"domain": "uniform", "low": 1e-4, "high": 1e-1},
        },
        "rf": {
            "n_estimators": {"domain": "loguniform", "low": 10, "high": 1000},
            "max_features": {"domain": "uniform", "low": 0.1, "high": 1.0},
        },
        "catboost": {
            "depth": {"domain": "uniform", "low": 4, "high": 10},
            "learning_rate": {"domain": "uniform", "low": 1e-4, "high": 1e-1},
        },
    },
}

运行 AutoML
automl.fit(X_train=X_train, y_train=y_train, **automl_settings)

输出结果
print("Best model:", automl.best_estimator)
print("Best hyperparameters:", automl.best_config)
print("Best accuracy:", automl.best_loss)

保存最佳模型
joblib.dump(automl.model, 'best_model.pkl')

加载最佳模型
best_model = joblib.load('best_model.pkl')

评估模型性能
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))





import optuna
from catboost import CatBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 加载数据
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义优化函数
def objective(trial):
    param = {
        'iterations': trial.suggest_int('iterations', 50, 300),  # 树的数量
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),  # 学习率
        'depth': trial.suggest_int('depth', 4, 10),  # 树的深度
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0),  # L2 正则化系数
        'border_count': trial.suggest_int('border_count', 50, 255),  # 分桶数（对于数值特征）
        'loss_function': 'Logloss',  # 优化的损失函数
        'eval_metric': 'Accuracy',  # 评估指标
        'random_seed': 42,  # 随机种子
        'verbose': False
    }

    model = CatBoostClassifier(**param)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=30)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return accuracy

# 创建一个 Optuna 学习器并运行优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# 输出最佳的参数和结果
print("Best trial:")
trial = study.best_trial
print(f"  Accuracy: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

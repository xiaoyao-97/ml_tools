import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

model = CatBoostRegressor()
model.fit(train[features], train[target], use_best_model=True)
pred = model.predict(test[features])
mean_squared_error(test[target], pred)

"""加一些params
params = {
    'objective': 'regression'
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 1,


    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3,
    'early_stopping_rounds': 50,
    'border_count': 32,
    'verbose': 100,
    'thread_count': -1
}
"""

"""参数的解释：
class CatBoostRegressor(iterations=None,
                        learning_rate=None,
                        depth=None,
                        l2_leaf_reg=None,
                        model_size_reg=None,
                        rsm=None,
                        loss_function='RMSE',
                        border_count=None,
                        feature_border_type=None,
                        per_float_feature_quantization=None,
                        input_borders=None,
                        output_borders=None,
                        fold_permutation_block=None,
                        od_pval=None,
                        od_wait=None,
                        od_type=None,
                        nan_mode=None,
                        counter_calc_method=None,
                        leaf_estimation_iterations=None,
                        leaf_estimation_method=None,
                        thread_count=None,
                        random_seed=None,
                        use_best_model=None,
                        best_model_min_trees=None,
                        verbose=None,
                        silent=None,
                        logging_level=None,
                        metric_period=None,
                        ctr_leaf_count_limit=None,
                        store_all_simple_ctr=None,
                        max_ctr_complexity=None,
                        has_time=None,
                        allow_const_label=None,
                        one_hot_max_size=None,
                        random_strength=None,
                        name=None,
                        ignored_features=None,
                        train_dir=None,
                        custom_metric=None,
                        eval_metric=None,
                        bagging_temperature=None,
                        save_snapshot=None,
                        snapshot_file=None,
                        snapshot_interval=None,
                        fold_len_multiplier=None,
                        used_ram_limit=None,
                        gpu_ram_part=None,
                        pinned_memory_size=None,
                        allow_writing_files=None,
                        final_ctr_computation_mode=None,
                        approx_on_full_history=None,
                        boosting_type=None,
                        simple_ctr=None,
                        combinations_ctr=None,
                        per_feature_ctr=None,
                        ctr_target_border_count=None,
                        task_type=None,
                        device_config=None,
                        devices=None,
                        bootstrap_type=None,
                        subsample=None,
                        sampling_unit=None,
                        dev_score_calc_obj_block_size=None,
                        max_depth=None,
                        n_estimators=None,
                        num_boost_round=None,
                        num_trees=None,
                        colsample_bylevel=None,
                        random_state=None,
                        reg_lambda=None,
                        objective=None,
                        eta=None,
                        max_bin=None,
                        gpu_cat_features_storage=None,
                        data_partition=None,
                        metadata=None,
                        early_stopping_rounds=None,
                        cat_features=None,
                        grow_policy=None,
                        min_data_in_leaf=None,
                        min_child_samples=None,
                        max_leaves=None,
                        num_leaves=None,
                        score_function=None,
                        leaf_estimation_backtracking=None,
                        ctr_history_unit=None,
                        monotone_constraints=None,
                        feature_weights=None,
                        penalties_coefficient=None,
                        first_feature_use_penalties=None,
                        model_shrink_rate=None,
                        model_shrink_mode=None,
                        langevin=None,
                        diffusion_temperature=None,
                        posterior_sampling=None,
                        boost_from_average=None,
                        fixed_binary_splits=None)


模型训练和树的配置
iterations (或 n_estimators, num_boost_round): 建树的数量。
learning_rate (或 eta): 学习步长，控制每次迭代更新的幅度。
depth (或 max_depth): 树的最大深度。
l2_leaf_reg (或 reg_lambda): L2 正则化项的系数，用于减少模型过拟合。
model_size_reg: 模型大小的正则化系数，控制模型的复杂度。
rsm: 训练每棵树时用于随机选择特征的比例。
boosting_type: 选择提升类型，通常是有序的（默认）或无序的。

损失函数和评价指标
loss_function: 使用的损失函数，如 'RMSE'。
eval_metric: 评价模型性能的指标。
custom_metric: 用户定义的额外评价指标。

特征处理
one_hot_max_size: 对于类别特征，超过此值的类别将被转换为更复杂的编码方式而不是独热编码。
cat_features: 指明哪些列是类别特征。
max_bin: 数值特征的最大分箱数量。
feature_weights: 给特征权重，影响训练过程中特征的选择。

正则化和复杂性控制
random_strength: 每个拆分所需的最小标准偏差的额外随机性。
bagging_temperature: 非对称抽样的温度参数，较高的值意味着更强的正则化。
leaf_estimation_method: 叶节点估计方法，比如梯度或牛顿。
leaf_estimation_iterations: 在叶估计过程中进行的迭代次数。

性能优化和硬件配置
thread_count: 使用的线程数量。
task_type: 指定是在 CPU 还是 GPU 上运行。
devices: 在 GPU 模式下使用的设备列表。

早停和模型保存
early_stopping_rounds: 如果设置，当验证集上的性能在指定轮数内没有提高，则停止训练。
save_snapshot: 是否保存训练过程中的快照，以便恢复训练。
snapshot_file: 快照文件的路径。
snapshot_interval: 保存快照的间隔。

其他
used_ram_limit: 训练过程中使用的最大 RAM。
boost_from_average: 是否从平均值开始提升，通常对对称损失函数有效。

特征处理和分箱
border_count: 用于数值特征的分箱时最大的不同边界数。
feature_border_type: 特征边界的类型，例如 GreedyLogSum, Uniform, MinEntropy 等。
per_float_feature_quantization: 指定每个浮点特征的量化配置。
input_borders: 手动指定特征的分界线。
output_borders: 手动指定模型输出的分界线。

数据分块和折叠
fold_permutation_block: 数据折叠的块大小，用于加速计算。
过拟合检测 (OD, Overfitting Detector)
od_pval: 过拟合检测器的 p 值阈值。
od_wait: 过拟合检测器等待的迭代数。
od_type: 过拟合检测类型，如 IncToDec 或 Iter。

缺失值和计数器
nan_mode: 处理缺失值的模式，如 Min, Max, Forbidden。
counter_calc_method: 计数器特征的计算方法。

树的生长和估计
leaf_estimation_iterations: 叶估计的迭代次数。
leaf_estimation_method: 叶估计方法，如 Newton, Gradient。
leaf_estimation_backtracking: 叶估计回溯类型，用于改进叶值。

性能和日志
thread_count: 线程数，用于并行处理。
logging_level: 日志级别，如 Silent, Verbose, Info。
metric_period: 指标计算的周期。
silent: 是否静默运行，不输出任何日志信息。

类别特征处理
ctr_leaf_count_limit: 类别特征转换的叶节点计数限制。
store_all_simple_ctr: 是否存储所有简单的类别特征转换结果。
max_ctr_complexity: 类别特征转换的最大复杂度。
ctr_target_border_count: 类别特征转换的目标边界数。
simple_ctr, combinations_ctr, per_feature_ctr: 不同类型的类别特征转换。

训练和保存模型
train_dir: 训练过程中保存模型和其他文件的目录。
save_snapshot: 是否保存训练快照。
snapshot_file: 快照文件的路径。
snapshot_interval: 保存快照的时间间隔。

资源管理
used_ram_limit: 使用的最大 RAM 限制。
gpu_ram_part, pinned_memory_size: GPU 相关的内存配置。

任务类型和设备
task_type: 任务类型，如 CPU 或 GPU。
device_config, devices: 指定使用的设备和配置。
Bootstrap 和采样
bootstrap_type: Bootstrap 类型，如 Bayesian, Bernoulli, MVS。
subsample: 子样本的比例，用于每棵树的训练。
sampling_unit: 采样单元的类型，影响数据分割。

策略和策略调整
grow_policy: 树的生长策略，如 SymmetricTree, Depthwise, Lossguide。
min_data_in_leaf, min_child_samples: 叶子节点最少的数据量。
max_leaves, num_leaves: 最大叶子数。

正则化和其他策略
random_strength: 增加训练中的随机性，以防止过拟合。
langevin, diffusion_temperature: 使用 Langevin 动力学进行随机梯度下降。
posterior_sampling: 后验采样，用于更准确的不确定性估计。
penalties_coefficient, first_feature_use_penalties: 特征使用的惩罚系数。

模型修正和约束
model_shrink_rate, model_shrink_mode: 模型收缩率和模式。
monotone_constraints: 单调约束，控制特征影响目标变量的方向。
boost_from_average: 是否从平均值开始提升，通常对对称损失函数有效。
feature_weights: 特征的权重。
"""

"""plot trees
brew install graphviz
model.plot_tree(tree_idx = 1)
"""

"""optuna+cv
import optuna
import numpy as np
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, Pool

# Generate some sample data
X, y = train[features].values, train[target]

def objective(trial):
    # Suggest values for the parameters we want to optimize
    param = {
        'objective': 'RMSE',
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 1,
        'iterations': trial.suggest_int('iterations', 3000, 8000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
        'depth': trial.suggest_int('depth', 5, 12),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1, 12),
        'border_count': trial.suggest_int('border_count', 82, 375),
        'verbose': False
    }

    # Cross-validation
    kfold = KFold(n_splits=5, shuffle=True) #, random_state=1)
    scores = []

    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = CatBoostRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)
        preds = model.predict(X_test)
        rmse = np.sqrt(np.mean((preds - y_test) ** 2))
        scores.append(rmse)

    # Calculate the average RMSE over all folds
    return np.mean(scores)

# Create a study object and specify the optimization direction
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)  # You can increase n_trials for more exhaustive search

# Print the best parameters
print('Best trial:', study.best_trial.params)
"""

"""hyperopt+cv
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

# Generate some sample data (replace with your actual data)
# X, y = train[features].values, train[target]

# Define the search space for the hyperparameters
space = {
    'iterations': hp.choice('iterations', np.arange(3000, 8000, step=100, dtype=int)),
    'learning_rate': hp.uniform('learning_rate', 0.005, 0.05),
    'depth': hp.choice('depth', np.arange(5, 13, step=1, dtype=int)),
    'l2_leaf_reg': hp.loguniform('l2_leaf_reg', np.log(1), np.log(12)),
    'border_count': hp.choice('border_count', np.arange(82, 376, step=1, dtype=int))
}

def objective(params):
    # Convert hyperparameters from float to integer as needed
    params['iterations'] = int(params['iterations'])
    params['depth'] = int(params['depth'])
    params['border_count'] = int(params['border_count'])

    # Cross-validation
    kfold = KFold(n_splits=5, shuffle=True) # Consider setting a random_state if needed
    scores = []

    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = CatBoostRegressor(
            objective='RMSE',
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=1,
            verbose=False,
            **params
        )

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        scores.append(rmse)

    # Calculate the average RMSE over all folds
    return {'loss': np.mean(scores), 'status': STATUS_OK}

# Run the algorithm
trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20,  # You can increase this number for a more exhaustive search
            trials=trials)

# Print the best parameters found
print("Best trial:", best)
"""





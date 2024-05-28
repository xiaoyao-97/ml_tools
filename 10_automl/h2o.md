pip install h2o

import h2o
h2o.init()


————————————————导入数据——————————————————————
data_path = "path_to_your_file.csv"
data = h2o.import_file(data_path)

————————————————查看数据——————————————————
data.head()
data.describe()
data[col].table()

———————————————————train test split—————————————————————
train, test = data.split_frame([0.8], seed=623)
print("train:%d test:%d" % (train.nrows, test.nrows))

data.names # columns

—————————————————————训练分类模型—————————————————————————————————
aml = H2OAutoML(max_models=25, 
                max_runtime_secs_per_model=10, 
                seed=623, 
                project_name='classification', 
                balance_classes=True, c
                lass_sampling_factors=[0.5,1.25])
%time aml.train(x=x, y=y, training_frame=train)

—————————————————————训练回归模型—————————————————————————————————————————————





输出结果：
model_ids = lb_df['model_id']
model_details = []
performance_metrics = []

for rank, row in lb_df.iterrows():
    model_id = row['model_id']
    model = h2o.get_model(model_id)
    model_details.append(model.params)
    performance = model.model_performance()
    performance_metrics.append({
        'model_id': model_id,
#         'auc': performance.auc(),
#         'logloss': performance.logloss(),
        'rmse': performance.rmse(),
#         'mse': performance.mse()
    })
    with open(f'h2o_1/{rank}_{model_id}_details.txt', 'w') as f:
        f.write(str(model.params))
    with open(f'h2o_1/{rank}_{model_id}_performance.txt', 'w') as f:
        f.write(str(performance))

预测
pred_h2o = aml.predict(test_h2o)

评估结果
np.sqrt(mse(test['p'],pred_h2o.as_data_frame()['predict']))

—————————————————————打印所有模型的performance———————————————————————————————————————
lb = aml.leaderboard
lb.head(rows=lb.nrows)


更详细的方式：
from h2o.automl import get_leaderboard
lb2 = get_leaderboard(aml, extra_columns='ALL')
lb2.head(rows=lb2.nrows)

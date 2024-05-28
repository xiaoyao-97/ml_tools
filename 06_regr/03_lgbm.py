"""
parameter tuning
https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html

LightGBM uses the leaf-wise tree growth algorithm, while many other popular tools use depth-wise tree growth. Compared with depth-wise growth, the leaf-wise algorithm can converge much faster. However, the leaf-wise growth may be over-fitting if not used with the appropriate parameters.
1. num_leaves. This is the main parameter to control the complexity of the tree model. Theoretically, we can set num_leaves = 2^(max_depth) to obtain the same number of leaves as depth-wise tree. However, this simple conversion is not good in practice. The reason is that a leaf-wise tree is typically much deeper than a depth-wise tree for a fixed number of leaves. Unconstrained depth can induce over-fitting. Thus, when trying to tune the num_leaves, we should let it be smaller than 2^(max_depth). For example, when the max_depth=7 the depth-wise tree can get good accuracy, but setting num_leaves to 127 may cause over-fitting, and setting it to 70 or 80 may get better accuracy than depth-wise.
2. min_data_in_leaf. This is a very important parameter to prevent over-fitting in a leaf-wise tree. Its optimal value depends on the number of training samples and num_leaves. Setting it to a large value can avoid growing too deep a tree, but may cause under-fitting. In practice, setting it to hundreds or thousands is enough for a large dataset.
3. max_depth. You also can use max_depth to limit the tree depth explicitly.



"""

# Convert the datasets into LightGBM dataset format
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Set up the parameters for LightGBM
params = {
    'boosting_type': 'gbdt',      # Gradient Boosting Decision Tree
    'objective': 'regression',    # Objective for regression tasks
    'metric': 'rmse',             # Root Mean Squared Error as the metric
    'num_leaves': 31,
    'learning_rate': 0.2,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Train the model
num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])

# Make predictions
y_pred = bst.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)  # Set squared=False to get the RMSE
print("RMSE:", rmse)







# Train the model
model = LGBMRegressor(**params)  
    
model.fit(X_train,y_train,eval_set=[(X_test,y_test)])

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)  # Set squared=False to get the RMSE
print("RMSE:", rmse)




# 另一个简单的版本：
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

params = {
    'objective': 'regression',
    'metric': 'rmse',
}

lgb_train = lgb.Dataset(X_train.drop("date_time",axis=1), y_no)

num_round = 1000
bst = lgb.train(params, lgb_train, num_round)

y_no_pred = bst.predict(X_test.drop("date_time",axis=1))


"""CV 重要！
X = train[features]
y = train[target]
test_tmp = test[features]
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def f_importance_plot(f_imp):
    fig = plt.figure(figsize = (15, 0.35*len(f_imp)))
    plt.title('Feature importances', size=25, y=1.05, 
              fontname='Calibri', fontweight='bold', color='#444444')
    a = sns.barplot(data=f_imp, x='avg_imp', y='feature', 
                    palette='Blues_d', linestyle="-", 
                    linewidth=1, edgecolor="black")
    plt.xlabel('')
    plt.xticks([])
    plt.ylabel('')
    plt.yticks(size=11, color='#444444')
    
    for j in ['right', 'top', 'bottom']:
        a.spines[j].set_visible(False)
    for j in ['left']:
        a.spines[j].set_linewidth(0.5)
    plt.show()
    
import gc
seed = 1
FOLDS =7
bold = ('\033[1m', '\033[0m')
lgb_params = {
    'max_depth': 9,
    'learning_rate': 0.01,
    'min_data_in_leaf': 36, 
    'num_leaves': 100, 
    'feature_fraction': 0.8, 
    'bagging_fraction': 0.89, 
    'bagging_freq': 5, 
    'lambda_l2': 28,
    
    'seed': seed,
    'objective': 'regression',
    'boosting_type': 'gbdt',
    'device': 'cpu', 
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'n_jobs': -1,
    'metric': 'rmse',
    'verbose': -1
}

f_imp = pd.DataFrame({'feature': X.columns})
predictions, scores = np.zeros(len(test)), []

from sklearn.model_selection import KFold
k = KFold(n_splits=FOLDS, random_state=seed, shuffle=True)
for fold, (train_idx, val_idx) in enumerate(k.split(X, y)):
    print(f'\n--- FOLD {fold+1} ---')
        
    lgb_train = lgb.Dataset(data=X.iloc[train_idx], 
                            label=y.iloc[train_idx],
                            categorical_feature=[])
    lgb_valid = lgb.Dataset(data=X.iloc[val_idx], 
                            label=y.iloc[val_idx],
                            categorical_feature=[],
                            reference=lgb_train)

    model = lgb.train(params=lgb_params, 
                      train_set=lgb_train, 
                      num_boost_round=50000,
                      valid_sets=[lgb_train, lgb_valid], 
                      valid_names=['train', 'val'],
                      callbacks=[lgb.log_evaluation(1000),
                                 lgb.early_stopping(1000, verbose=False)])
    
    f_imp['fold_'+str(fold+1)] = model.feature_importance()
    b_itr = model.best_iteration
    
    val_preds = model.predict(X.iloc[val_idx], num_iteration=b_itr)
    val_score = rmse(y.iloc[val_idx], val_preds)
    scores.append(val_score)
    
    predictions += model.predict(test_tmp, num_iteration=b_itr) / FOLDS
    print(f'--- RMSE: {bold[0]}{round(val_score, 6)}{bold[1]} | best iteration: {bold[0]}{b_itr}{bold[1]} ---')
    
    del lgb_train, lgb_valid, val_preds, val_score, model
    gc.collect()

print('*'*45)
print(f'Mean RMSE: {bold[0]}{round(np.mean(scores), 6)}{bold[1]}')

f_imp['avg_imp'] = f_imp[f_imp.columns[1:]].mean(axis=1)
f_imp.sort_values('avg_imp', ascending=False, inplace=True)
f_importance_plot(f_imp)"""
#—————————————————————————————————————————————————————————————————————
"""简单的
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMRegressor(
    boosting_type='gbdt',
    objective='regression',
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=100
)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)])

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse:.4f}')
"""





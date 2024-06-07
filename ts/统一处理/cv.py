import pandas as pd

def ts_cv(df, start_point, fold=None, test_length=None, train_length=None):
    n = len(df)
    
    if test_length is None:
        test_length = (n - start_point) // fold if fold else (n - start_point) // 5
    
    if fold is None:
        fold = (n - start_point) // test_length
    
    step = (n - start_point - test_length) // (fold - 1) if fold > 1 else (n - start_point)
    
    folds = []
    
    for i in range(fold):
        if i == fold - 1:
            test_start = n - test_length
        else:
            test_start = start_point + i * step
        test_end = test_start + test_length
        
        if test_end > n:
            test_end = n
        
        if train_length is None:
            train_end = test_start
            train_start = 0
        else:
            train_end = test_start
            train_start = max(0, train_end - train_length)
        
        train_index = list(range(train_start, train_end))
        test_index = list(range(test_start, test_end))
        
        folds.append((train_index, test_index))
    
    return folds


# df = pd.DataFrame({'date': pd.date_range(start='1/1/2020', periods=150), 'value': range(150)})
# folds = ts_cv(df, start_point=50, fold=10, train_length=None, test_length=30)
# for train_index, test_index in folds:
#     print("Train indices:", train_index)
#     print("Test indices:", test_index)


# random
import pandas as pd
import numpy as np

def ts_cv_random(df, start_point, fold=None, test_length=None, train_length=None, random_state=None, ratio=0.8):
    np.random.seed(random_state)
    n = len(df)
    
    if test_length is None:
        test_length = (n - start_point) // fold if fold else (n - start_point) // 5
    
    if fold is None:
        fold = (n - start_point) // test_length
    
    step = (n - start_point - test_length) // (fold - 1) if fold > 1 else (n - start_point)
    
    folds = []
    
    for i in range(fold):
        if i == fold - 1:
            test_start = n - test_length
        else:
            test_start = start_point + i * step
        test_end = test_start + test_length
        
        if test_end > n:
            test_end = n
        
        if train_length is None:
            train_end = test_start
            train_start = 0
        else:
            train_end = test_start
            train_start = max(0, train_end - train_length)
        
        train_index = list(range(train_start, train_end))
        test_index = list(range(test_start, test_end))
        
        train_index = np.random.choice(train_index, size=int(len(train_index) * ratio), replace=False).tolist()
        test_index = np.random.choice(test_index, size=int(len(test_index) * ratio), replace=False).tolist()
        
        folds.append((train_index, test_index))
    
    return folds

# 示例使用方法
df = pd.DataFrame({'date': pd.date_range(start='1/1/2020', periods=150), 'value': range(150)})
folds = ts_cv_random(df, start_point=50, fold=10, train_length=40, test_length=50, random_state=42, ratio=0.8)
for train_index, test_index in folds:
    print("Train indices:", train_index)
    print("Test indices:", test_index)






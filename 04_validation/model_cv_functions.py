"""model to f1 prediction
def f1_lr(X_train, y_train, X_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    return y_train_pred, y_test_pred
"""

# loss function
def f_loss(y_test, y_pred):
    return np.array(y_test)@np.array(y_pred)/len(y_test)

"""cv f2 without loss
from sklearn.model_selection import KFold

def f2(model_func, X_train, y_train, X_test, cv=5, random_state = 1):
    
    kf = KFold(n_splits = cv, shuffle = True, random_state = random_state)

    oof_preds = np.zeros(X_train.shape[0])
    
    for train_index, test_index in kf.split(X_train):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        
        y_train_fold_pred, y_test_fold_pred = model_func(X_train_fold, y_train_fold, X_test_fold)
        
        oof_preds[test_index] = y_test_fold_pred

    y_train_pred_full, y_test_pred_full = model_func(X_train, y_train, X_test)
    
    return oof_preds, y_train_pred_full, y_test_pred_full"""

"""cv f2 with loss
from sklearn.model_selection import KFold
import numpy as np

def f2(model_func, X_train, y_train, X_test, f_loss=None, cv=5, random_state=1):
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(X_train.shape[0])
    losses = []
    
    for train_index, test_index in kf.split(X_train):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        
        y_train_fold_pred, y_test_fold_pred = model_func(X_train_fold, y_train_fold, X_test_fold)
        
        oof_preds[test_index] = y_test_fold_pred
        
        if f_loss is not None:
            loss = f_loss(y_test_fold, y_test_fold_pred)
            losses.append(loss)
    
    y_train_pred_full, y_test_pred_full = model_func(X_train, y_train, X_test)
    
    total_loss = f_loss(y_train, oof_preds) if f_loss is not None else None
    average_loss = np.mean(losses) if losses else None
    
    return {
        'oof_preds': oof_preds,
        'losses': losses,
        'average_loss': average_loss,
        'total_loss': total_loss,
        'y_train_pred_full': y_train_pred_full,
        'y_test_pred_full': y_test_pred_full
    }
"""



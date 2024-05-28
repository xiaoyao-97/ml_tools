# 给 cv score 和 feature importance
def cv_feat_imp(model, df, features, target, cv_fold=5, random_state=1):
    kf = KFold(n_splits=cv_fold, shuffle=True, random_state=random_state)
    losses = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        loss = mean_squared_error(y_val, preds)
        losses.append(loss)
        
        print(f'Fold {fold + 1} Loss: {loss}')
    
    avg_loss = np.mean(losses)
    print(f'Average Loss: {avg_loss}')
    
    model.fit(df[features], df[target])
    feature_importances = model.feature_importances_
    
    feat_imp_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    plt.figure(figsize=(5, 3))
    plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    plt.gca().invert_yaxis()
    plt.show()
    
    return avg_loss, feat_imp_df


# 逐次移除features
def feli_feat_imp(model, df, features, target, step, early_stop=5, cv_fold=5, random_state=1):
    remaining_features = features.copy()
    history = []
    no_improvement_count = 0
    best_loss = float('inf')

    while len(remaining_features) > step and no_improvement_count < early_stop:
        avg_loss, feat_imp_df = cv_feat_imp(model, df, remaining_features, target, cv_fold, random_state)
        
        history.append((remaining_features.copy(), feat_imp_df.copy(), avg_loss))
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        least_important_features = feat_imp_df.tail(step)['Feature'].tolist()
        remaining_features = [feat for feat in remaining_features if feat not in least_important_features]
    
    num_features_list = [len(h[0]) for h in history]
    losses_list = [h[2] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(num_features_list, losses_list, marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Loss (MSE)')
    plt.title('Feature Elimination Impact on Loss')
    plt.gca().invert_xaxis()
    plt.show()
    
    return history


# 只和last 比较
def feli_feat_imp(model, df, features, target, step, early_stop=5, cv_fold=5, random_state=1):
    remaining_features = features.copy()
    history = []
    no_improvement_count = 0
    last_loss = float('inf')

    while len(remaining_features) > step and no_improvement_count < early_stop:
        avg_loss, feat_imp_df = cv_feat_imp(model, df, remaining_features, target, cv_fold, random_state)
        
        history.append((remaining_features.copy(), feat_imp_df.copy(), avg_loss))
        
        if avg_loss < last_loss:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        last_loss = avg_loss
        
        least_important_features = feat_imp_df.tail(step)['Feature'].tolist()
        remaining_features = [feat for feat in remaining_features if feat not in least_important_features]
    
    num_features_list = [len(h[0]) for h in history]
    losses_list = [h[2] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(num_features_list, losses_list, marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Loss (MSE)')
    plt.title('Feature Elimination Impact on Loss')
    plt.gca().invert_xaxis()
    plt.show()
    
    return history


# 











# cv输出loss和shap_df
def cv_shap(model, df, features, target, cv_fold=5, random_state=1):
    kf = KFold(n_splits=cv_fold, shuffle=True, random_state=random_state)
    losses = []
    shap_values_list = []
    base_values_list = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        loss = mean_squared_error(y_val, preds)
        losses.append(loss)
        
        print(f'Fold {fold + 1} Loss: {loss}')
        
        # Calculate SHAP values
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_val)
        shap_values_list.append(shap_values)
        base_values_list.append(shap_values.base_values)

    avg_loss = np.mean(losses)
    print(f'Average Loss: {avg_loss}')
    
    shap_values_combined = np.vstack([shap_values.values for shap_values in shap_values_list])
    shap_mean_importance = np.abs(shap_values_combined).mean(axis=0)

    shap_df = pd.DataFrame({
        'Feature': features,
        'Importance': shap_mean_importance
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    plt.figure()
    shap.summary_plot(np.vstack([sv.values for sv in shap_values_list]), np.vstack([sv.data for sv in shap_values_list]), feature_names=features)
    
    plt.figure()
    shap.summary_plot(np.vstack([sv.values for sv in shap_values_list]), np.vstack([sv.data for sv in shap_values_list]), feature_names=features, plot_type='bar')

    return avg_loss, shap_df


# def feli_shap(model, df, features, target, step, early_stop=5, cv_fold=5, random_state=1):
    remaining_features = features.copy()
    history = []
    no_improvement_count = 0
    best_loss = float('inf')

    while len(remaining_features) > step and no_improvement_count < early_stop:
        avg_loss, feat_imp_df = cv_shap(model, df, remaining_features, target, cv_fold, random_state)
        
        history.append((remaining_features.copy(), feat_imp_df.copy(), avg_loss))
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Remove the least important features based on SHAP values
        least_important_features = feat_imp_df.tail(step)['Feature'].tolist()
        remaining_features = [feat for feat in remaining_features if feat not in least_important_features]
    
    num_features_list = [len(h[0]) for h in history]
    losses_list = [h[2] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(num_features_list, losses_list, marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Loss (MSE)')
    plt.title('Feature Elimination Impact on Loss using SHAP')
    plt.gca().invert_xaxis()
    plt.show()
    
    return history



# 

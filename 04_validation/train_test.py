X = train.drop(columns = ['target']).values
y = train.target.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333333333333, random_state=1)
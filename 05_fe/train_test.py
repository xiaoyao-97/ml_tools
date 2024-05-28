import pandas as pd

def process_features(train, test):
    target = train['target']
    train = train.drop(columns=['target'])
    combined = pd.concat([train, test], keys=['train', 'test'])
    # TODO
    train_processed = new.xs('train')
    test_processed = new.xs('test')
    train_processed['target'] = target

    return train_processed, test_processed

def process_features(train, test):
    l1 = len(train)
    target = train['target']
    train = train.drop(columns=['target'])
    combined = pd.concat([train, test], ignore_index=True)
    # TODO
    train_processed = new[:l1]
    test_processed = new[l1:]
    train_processed['target'] = target

    return train_processed, test_processed

targets = ['target_carbon_monoxide', 'target_benzene', 'target_nitrogen_oxides']
train_features = train.drop(columns=targets)
train_features['type'] = 'train'
test['type'] = 'test'
data = pd.concat([train_features, test], axis=0)
data.reset_index(drop=True, inplace=True)
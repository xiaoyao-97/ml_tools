"""pickle
import pickle

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
"""

""" joblib
from joblib import dump

dump(model, 'model.joblib')

from joblib import load

model = load('model.joblib')
"""

"""catb
model.save_model('model.cbm', format="cbm")

from catboost import CatBoostRegressor
model = CatBoostRegressor()
model.load_model('model.cbm', format='cbm')  # 指定格式
"""




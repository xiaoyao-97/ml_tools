使用session_id的一个重要作用：确保结果的可重复性和一致性

直接创造对象
from pycaret.classification import *
s = setup(data, target = 'Class variable', session_id = 123)

先创造一个类的对象，然后对这个对象赋值
from pycaret.classification import ClassificationExperiment
s = ClassificationExperiment()
s.setup(data, target = 'Class variable', session_id = 123)

———————————————————————————比较模型—————————————————————————————————
# functional API
best = compare_models()

# OOP API
best = s.compare_models()

——————————————————————————模型的预测——————————————————————————————
对session的test:
predict_model(best)

对任意数据的预测：
predictions = s.predict_model(best, data=data)
predictions.head()

————————————————————————————模型的储存和提取————————————————————————————————————
s.save_model(best, 'my_best_pipeline')


loaded_model = s.load_model('my_best_pipeline')
print(loaded_model)




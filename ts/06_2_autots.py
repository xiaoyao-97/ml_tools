


"""基本代码
from autots import AutoTS

model = AutoTS(
    forecast_length=10,  # 预测未来 10 天
    frequency='infer',   # 自动推断时间序列频率
    prediction_interval=0.9  # 90% 预测区间
)

model = model.fit(
    df, 
    date_col='date', 
    value_col='value', 
    id_col=None
    )

prediction = model.predict()
forecast = prediction.forecast

from sklearn.metrics import mean_squared_error

y_true = df['value'][-10:].values
y_pred = forecast['value'][:10].values

mse = mean_squared_error(y_true, y_pred)
print("MSE on validation set:", mse)"""


"""提高精度：
max_generations=15 # 简单的提高精度的做法
ensemble='all'

no_negatives=True # 数据非负
constraint=2.0. 限制预测的范围： max(training data) + 2.0 * st.dev(training data)

drop_most_recent=1 # 不管最近的
subset=100 # 在多少个时间序列上测试模型

num_validations

model_list = {'probabilistic', 'multivariate', 'fast', 'superfast', or 'all'}
    On large multivariate series, DynamicFactor and VARMAX can be impractically slow.
"""

"""running just one model:
from autots import load_daily, model_forecast
df = load_daily(long=False)  # long or non-numeric data won't work with this function
df_forecast = model_forecast(
    model_name="AverageValueNaive",
    model_param_dict={'method': 'Mean'},
    model_transform_dict={
        'fillna': 'mean',
        'transformations': {'0': 'DifferencedTransformer'},
        'transformation_params': {'0': {}}
    },
    df_train=df,
    forecast_length=12,
    frequency='infer',
    prediction_interval=0.9,
    no_negatives=False,
    # future_regressor_train=future_regressor_train2d,
    # future_regressor_forecast=future_regressor_forecast2d,
    random_seed=321,
    verbose=0,
    n_jobs="auto",
)
df_forecast.forecast.head(5)"""

"""metric_weighting = {
    'smape_weighting': 5,
    'mae_weighting': 2,
    'rmse_weighting': 2,
    'made_weighting': 0.5,
    'mage_weighting': 1,
    'mle_weighting': 0,
    'imle_weighting': 0,
    'spl_weighting': 3,
    'containment_weighting': 0,
    'contour_weighting': 1,
    'runtime_weighting': 0.05,
}

model = AutoTS(
    forecast_length=forecast_length,
    frequency='infer',
    metric_weighting=metric_weighting,
)"""

"""AutoTS的文档
Args:
    forecast_length (int): number of periods over which to evaluate forecast. Can be overriden later in .predict().
        when you don't have much historical data, using a small forecast length for .fit and the full desired forecast lenght for .predict is usually the best possible approach given limitations.
    frequency (str): 'infer' or a specific pandas datetime offset. Can be used to force rollup of data (ie daily input, but frequency 'M' will rollup to monthly).
    prediction_interval (float): 0-1, uncertainty range for upper and lower forecasts. Adjust range, but rarely matches actual containment.
    max_generations (int): number of genetic algorithms generations to run.
        More runs = longer runtime, generally better accuracy.
        It's called `max` because someday there will be an auto early stopping option, but for now this is just the exact number of generations to run.
    no_negatives (bool): if True, all negative predictions are rounded up to 0.
    constraint (float): when not None, use this float value * data st dev above max or below min for constraining forecast values.
        now also instead accepts a dictionary containing the following key/values:
            constraint_method (str): one of
                stdev_min - threshold is min and max of historic data +/- constraint * st dev of data
                stdev - threshold is the mean of historic data +/- constraint * st dev of data
                absolute - input is array of length series containing the threshold's final value for each
                quantile - constraint is the quantile of historic data to use as threshold
            constraint_regularization (float): 0 to 1
                where 0 means no constraint, 1 is hard threshold cutoff, and in between is penalty term
            upper_constraint (float): or array, depending on method, None if unused
            lower_constraint (float): or array, depending on method, None if unused
            bounds (bool): if True, apply to upper/lower forecast, otherwise False applies only to forecast
    ensemble (str): None or list or comma-separated string containing:
        'auto', 'simple', 'distance', 'horizontal', 'horizontal-min', 'horizontal-max', "mosaic", "subsample"
    initial_template (str): 'Random' - randomly generates starting template, 'General' uses template included in package, 'General+Random' - both of previous. Also can be overriden with self.import_template()
    random_seed (int): random seed allows (slightly) more consistent results.
    holiday_country (str): passed through to Holidays package for some models.
    subset (int): maximum number of series to evaluate at once. Useful to speed evaluation when many series are input.
        takes a new subset of columns on each validation, unless mosaic ensembling, in which case columns are the same in each validation
    aggfunc (str): if data is to be rolled up to a higher frequency (daily -> monthly) or duplicate timestamps are included. Default 'first' removes duplicates, for rollup try 'mean' or np.sum.
        Beware numeric aggregations like 'mean' will not work with non-numeric inputs.
        Numeric aggregations like 'sum' will also change nan values to 0
    na_tolerance (float): 0 to 1. Series are dropped if they have more than this percent NaN. 0.95 here would allow series containing up to 95% NaN values.
    metric_weighting (dict): weights to assign to metrics, effecting how the ranking score is generated.
    drop_most_recent (int): option to drop n most recent data points. Useful, say, for monthly sales data where the current (unfinished) month is included.
        occurs after any aggregration is applied, so will be whatever is specified by frequency, will drop n frequencies
    drop_data_older_than_periods (int): take only the n most recent timestamps
    model_list (list): str alias or list of names of model objects to use
        now can be a dictionary of {"model": prob} but only affects starting random templates. Genetic algorithim takes from there.
    transformer_list (list): list of transformers to use, or dict of transformer:probability. Note this does not apply to initial templates.
        can accept string aliases: "all", "fast", "superfast", 'scalable' (scalable is a subset of fast that should have fewer memory issues at scale)
    transformer_max_depth (int): maximum number of sequential transformers to generate for new Random Transformers. Fewer will be faster.
    models_mode (str): option to adjust parameter options for newly generated models. Only sporadically utilized. Currently includes:
        'default'/'random', 'deep' (searches more params, likely slower), and 'regressor' (forces 'User' regressor mode in regressor capable models),
        'gradient_boosting', 'neuralnets' (~Regression class models only)
    num_validations (int): number of cross validations to perform. 0 for just train/test on best split.
        Possible confusion: num_validations is the number of validations to perform *after* the first eval segment, so totally eval/validations will be this + 1.
        Also "auto" and "max" aliases available. Max maxes out at 50.
    models_to_validate (int): top n models to pass through to cross validation. Or float in 0 to 1 as % of tried.
        0.99 is forced to 100% validation. 1 evaluates just 1 model.
        If horizontal or mosaic ensemble, then additional min per_series models above the number here are added to validation.
    max_per_model_class (int): of the models_to_validate what is the maximum to pass from any one model class/family.
    validation_method (str): 'even', 'backwards', or 'seasonal n' where n is an integer of seasonal
        'backwards' is better for recency and for shorter training sets
        'even' splits the data into equally-sized slices best for more consistent data, a poetic but less effective strategy than others here
        'seasonal' most similar indexes
        'seasonal n' for example 'seasonal 364' would test all data on each previous year of the forecast_length that would immediately follow the training data.
        'similarity' automatically finds the data sections most similar to the most recent data that will be used for prediction
        'custom' - if used, .fit() needs validation_indexes passed - a list of pd.DatetimeIndex's, tail of each is used as test
    min_allowed_train_percent (float): percent of forecast length to allow as min training, else raises error.
        0.5 with a forecast length of 10 would mean 5 training points are mandated, for a total of 15 points.
        Useful in (unrecommended) cases where forecast_length > training length.
    remove_leading_zeroes (bool): replace leading zeroes with NaN. Useful in data where initial zeroes mean data collection hasn't started yet.
    prefill_na (str): value to input to fill all NaNs with. Leaving as None and allowing model interpolation is recommended.
        None, 0, 'mean', or 'median'. 0 may be useful in for examples sales cases where all NaN can be assumed equal to zero.
    introduce_na (bool): whether to force last values in one training validation to be NaN. Helps make more robust models.
        defaults to None, which introduces NaN in last rows of validations if any NaN in tail of training data. Will not introduce NaN to all series if subset is used.
        if True, will also randomly change 20% of all rows to NaN in the validations
    preclean (dict): if not None, a dictionary of Transformer params to be applied to input data
        {"fillna": "median", "transformations": {}, "transformation_params": {}}
        This will change data used in model inputs for fit and predict, and for accuracy evaluation in cross validation!
    model_interrupt (bool): if False, KeyboardInterrupts quit entire program.
        if True, KeyboardInterrupts attempt to only quit current model.
        if True, recommend use in conjunction with `verbose` > 0 and `result_file` in the event of accidental complete termination.
        if "end_generation", as True and also ends entire generation of run. Note skipped models will not be tried again.
    generation_timeout (int): if not None, this is the number of minutes from start at which the generational search ends, then proceeding to validation
        This is only checked after the end of each generation, so only offers an 'approximate' timeout for searching. It is an overall cap for total generation search time, not per generation.
    current_model_file (str): file path to write to disk of current model params (for debugging if computer crashes). .json is appended
    force_gc (bool): if True, run gc.collect() after each model run. Probably won't make much difference.
    horizontal_ensemble_validation (bool): True is slower but more reliable model selection on unstable data, if horz. ensembles are used
    verbose (int): setting to 0 or lower should reduce most output. Higher numbers give more output.
    n_jobs (int): Number of cores available to pass to parallel processing. A joblib context manager can be used instead (pass None in this case). Also 'auto'.

Attributes:
    best_model (pd.DataFrame): DataFrame containing template for the best ranked model
    best_model_name (str): model name
    best_model_params (dict): model params
    best_model_transformation_params (dict): transformation parameters
    best_model_ensemble (int): Ensemble type int id
    regression_check (bool): If True, the best_model uses an input 'User' future_regressor
    df_wide_numeric (pd.DataFrame): dataframe containing shaped final data, will include preclean
    initial_results.model_results (object): contains a collection of result metrics
    score_per_series (pd.DataFrame): generated score of metrics given per input series, if horizontal ensembles

Methods:
    fit, predict
    export_template, import_template, import_results, import_best_model
    results, failure_rate
    horizontal_to_df, mosaic_to_df
    plot_horizontal, plot_horizontal_transformers, plot_generation_loss, plot_backforecast
"""

"""加快速度：
model_list = ["superfast", "fast", "fast_parallel"]
from autots.models.model_list import model_lists
最重要的就是把不费时但是精度高的模型记录下来。
"""

"""
Model Number: 1 of 67 with model Ensemble for Validation 3
📈 1 - Ensemble with avg smape 11.19: 
Model Number: 2 of 67 with model Ensemble for Validation 3
2 - Ensemble with avg smape 11.38: 
Model Number: 3 of 67 with model Ensemble for Validation 3
3 - Ensemble with avg smape 12.74: 
Model Number: 4 of 67 with model Ensemble for Validation 3
4 - Ensemble with avg smape 12.69: 
Model Number: 5 of 67 with model MultivariateMotif for Validation 3
5 - MultivariateMotif with avg smape 14.43: 
Model Number: 6 of 67 with model LastValueNaive for Validation 3
6 - LastValueNaive with avg smape 14.32: 
Model Number: 7 of 67 with model LastValueNaive for Validation 3
7 - LastValueNaive with avg smape 14.16: 
Model Number: 8 of 67 with model LastValueNaive for Validation 3
8 - LastValueNaive with avg smape 14.18: 
Model Number: 9 of 67 with model LastValueNaive for Validation 3
9 - LastValueNaive with avg smape 14.16: 
Model Number: 10 of 67 with model NVAR for Validation 3
📈 10 - NVAR with avg smape 10.6: 
Model Number: 11 of 67 with model MultivariateMotif for Validation 3
11 - MultivariateMotif with avg smape 12.98: 
Model Number: 12 of 67 with model MultivariateMotif for Validation 3
12 - MultivariateMotif with avg smape 13.76: 
Model Number: 13 of 67 with model SectionalMotif for Validation 3
13 - SectionalMotif with avg smape 13.48: 
Model Number: 14 of 67 with model UnivariateMotif for Validation 3
14 - UnivariateMotif with avg smape 15.51: 
Model Number: 15 of 67 with model UnivariateMotif for Validation 3
15 - UnivariateMotif with avg smape 12.69: 
Model Number: 16 of 67 with model MultivariateMotif for Validation 3
16 - MultivariateMotif with avg smape 11.88: 
Model Number: 17 of 67 with model Theta for Validation 3
17 - Theta with avg smape 12.49: 
Model Number: 18 of 67 with model Theta for Validation 3
18 - Theta with avg smape 12.92: 
Model Number: 19 of 67 with model UnivariateMotif for Validation 3
19 - UnivariateMotif with avg smape 14.79: 
Model Number: 20 of 67 with model VAR for Validation 3
20 - VAR with avg smape 13.19: 
Model Number: 21 of 67 with model SeasonalityMotif for Validation 3
21 - SeasonalityMotif with avg smape 14.92: 
Model Number: 22 of 67 with model SectionalMotif for Validation 3
22 - SectionalMotif with avg smape 13.59: 
Model Number: 23 of 67 with model UnivariateMotif for Validation 3
23 - UnivariateMotif with avg smape 14.1: 
Model Number: 24 of 67 with model ETS for Validation 3
24 - ETS with avg smape 14.16: 
Model Number: 25 of 67 with model AverageValueNaive for Validation 3
25 - AverageValueNaive with avg smape 12.98: 
Model Number: 26 of 67 with model SectionalMotif for Validation 3
26 - SectionalMotif with avg smape 12.84: 
Model Number: 27 of 67 with model SeasonalNaive for Validation 3
27 - SeasonalNaive with avg smape 15.89: 
Model Number: 28 of 67 with model Theta for Validation 3
28 - Theta with avg smape 14.26: 
Model Number: 29 of 67 with model UnobservedComponents for Validation 3
29 - UnobservedComponents with avg smape 14.12: 
Model Number: 30 of 67 with model Theta for Validation 3
30 - Theta with avg smape 14.43: 
Model Number: 31 of 67 with model ConstantNaive for Validation 3
31 - ConstantNaive with avg smape 14.1: 
Model Number: 32 of 67 with model ConstantNaive for Validation 3
32 - ConstantNaive with avg smape 14.18: 
Model Number: 33 of 67 with model SectionalMotif for Validation 3
Template Eval Error: Exception("Transformer DatepartRegression failed on inverse from params ffill {'0': {}, '1': {'regression_model': {'model': 'DecisionTree', 'model_params': {'max_depth': None, 'min_samples_split': 1.0}}, 'datepart_method': 'expanded', 'polynomial_degree': None, 'transform_dict': None, 'holiday_countries_used': True}, '2': {'rows': 1, 'lag': 1, 'method': 'additive', 'strength': 1.0, 'first_value_only': False}, '3': {'window': 364}}") in model 33 in generation 0: SectionalMotif
Model Number: 34 of 67 with model VAR for Validation 3
34 - VAR with avg smape 15.79: 
Model Number: 35 of 67 with model ConstantNaive for Validation 3
35 - ConstantNaive with avg smape 14.06: 
Model Number: 36 of 67 with model MultivariateRegression for Validation 3
36 - MultivariateRegression with avg smape 14.47: 
Model Number: 37 of 67 with model MetricMotif for Validation 3
37 - MetricMotif with avg smape 23.9: 
Model Number: 38 of 67 with model VAR for Validation 3
38 - VAR with avg smape 18.12: 
Model Number: 39 of 67 with model UnobservedComponents for Validation 3
39 - UnobservedComponents with avg smape 14.16: 
Model Number: 40 of 67 with model ConstantNaive for Validation 3
40 - ConstantNaive with avg smape 14.18: 
Model Number: 41 of 67 with model GLS for Validation 3
41 - GLS with avg smape 15.11: 
Model Number: 42 of 67 with model SeasonalityMotif for Validation 3
42 - SeasonalityMotif with avg smape 12.06: 
Model Number: 43 of 67 with model SeasonalityMotif for Validation 3
43 - SeasonalityMotif with avg smape 14.57: 
Model Number: 44 of 67 with model UnobservedComponents for Validation 3
44 - UnobservedComponents with avg smape 14.1: 
Model Number: 45 of 67 with model SeasonalityMotif for Validation 3
45 - SeasonalityMotif with avg smape 17.81: 
Model Number: 46 of 67 with model AverageValueNaive for Validation 3
46 - AverageValueNaive with avg smape 14.71: 
Model Number: 47 of 67 with model GLM for Validation 3
47 - GLM with avg smape 15.94: 
Model Number: 48 of 67 with model UnivariateRegression for Validation 3
48 - UnivariateRegression with avg smape 14.18: 
Model Number: 49 of 67 with model FBProphet for Validation 3
16:40:56 - cmdstanpy - INFO - Chain [1] start processing
16:40:56 - cmdstanpy - INFO - Chain [1] start processing
16:40:56 - cmdstanpy - INFO - Chain [1] done processing
16:40:56 - cmdstanpy - INFO - Chain [1] done processing
16:40:57 - cmdstanpy - INFO - Chain [1] start processing
16:40:57 - cmdstanpy - INFO - Chain [1] done processing
49 - FBProphet with avg smape 14.18: 
Model Number: 50 of 67 with model ETS for Validation 3
50 - ETS with avg smape 28.83: 
Model Number: 51 of 67 with model NVAR for Validation 3
51 - NVAR with avg smape 16.25: 
Model Number: 52 of 67 with model NVAR for Validation 3
52 - NVAR with avg smape 15.54: 
Model Number: 53 of 67 with model UnobservedComponents for Validation 3
53 - UnobservedComponents with avg smape 17.62: 
Model Number: 54 of 67 with model VAR for Validation 3
54 - VAR with avg smape 13.02: 
Model Number: 55 of 67 with model SeasonalNaive for Validation 3
55 - SeasonalNaive with avg smape 14.18: 
Model Number: 56 of 67 with model SeasonalNaive for Validation 3
56 - SeasonalNaive with avg smape 14.18: 
Model Number: 57 of 67 with model MetricMotif for Validation 3
57 - MetricMotif with avg smape 15.81: 
Model Number: 58 of 67 with model AverageValueNaive for Validation 3
58 - AverageValueNaive with avg smape 13.48: 
Model Number: 59 of 67 with model MultivariateRegression for Validation 3
59 - MultivariateRegression with avg smape 15.23: 
Model Number: 60 of 67 with model AverageValueNaive for Validation 3
60 - AverageValueNaive with avg smape 14.35: 
Model Number: 61 of 67 with model MultivariateRegression for Validation 3
61 - MultivariateRegression with avg smape 19.81: 
Model Number: 62 of 67 with model WindowRegression for Validation 3
62 - WindowRegression with avg smape 13.81: 
Model Number: 63 of 67 with model VECM for Validation 3
63 - VECM with avg smape 26.24: 
Model Number: 64 of 67 with model VECM for Validation 3
64 - VECM with avg smape 26.23: 
Model Number: 65 of 67 with model MultivariateRegression for Validation 3
65 - MultivariateRegression with avg smape 11.72: 
Model Number: 66 of 67 with model VECM for Validation 3
66 - VECM with avg smape 17.02: 
Model Number: 67 of 67 with model FBProphet for Validation 3
67 - FBProphet with avg smape 16.33: """

model.best_model_name,
model.best_model_params,
model.best_model_transformation_params

model.plot_per_series_mape(kind="pie")
plt.show()

model.plot_per_series_error()
plt.show()

model.plot_generation_loss()
plt.show()

if model.best_model_ensemble == 2:
    model.plot_horizontal_per_generation()
    plt.show()
    model.plot_horizontal_transformers(method="fillna")
    plt.show()
    model.plot_horizontal_transformers()
    plt.show()
    model.plot_horizontal()
    plt.show()
    if "mosaic" in model.best_model["ModelParameters"].iloc[0].lower():
        mosaic_df = model.mosaic_to_df()
        print(mosaic_df[mosaic_df.columns[0:5]].head(5))

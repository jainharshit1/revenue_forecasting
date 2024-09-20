import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from transform import TimeSeriesDifferencer, SquareRootTransformer, SimpleMovingAverage, EWMAOptimizer, \
    BoxCoxTransformer, CustomStandardScaler, ARIMATransformer, RawData
from model_leads import ARIMASARIMAXModel, ProphetModel, ARIMASARIMAXModel_
from statsmodels.tsa.stattools import adfuller, kpss
from prophet import Prophet
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.filters.hp_filter import hpfilter

data = pd.read_csv('leads_report_df_989699.csv')
dataset = data[['email', 'created_at']].copy()
dataset['date'] = dataset['created_at'].str.split(' ').str[0]
dataset.drop('created_at', axis=1, inplace=True)
dataset['date'] = pd.to_datetime(dataset['date'])
dataset['day'] = 1
dataset['month'] = dataset['date'].dt.month
dataset['year'] = dataset['date'].dt.year
dataset['new_date'] = pd.to_datetime(dataset[['year', 'month', 'day']])
grouped_data = dataset.groupby('new_date').agg({'email': lambda x: ', '.join(x), 'day': 'count'}).rename(
    columns={'day': 'count'}).reset_index()
grouped_data = grouped_data.drop('email', axis=1)
grouped_data = grouped_data.drop(grouped_data.index[-1])
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
grouped_data_1 = grouped_data[168:173]
grouped_data_pycaret = grouped_data.copy()
grouped_data = grouped_data[141:169]
grouped_data['count_orignal']=grouped_data['count']
count_sqrt=np.array(grouped_data['count'])
sqrt_transformed = np.sqrt(count_sqrt)
grouped_data['count']=sqrt_transformed
array_ar=np.array(grouped_data['count'])
decomposition = sm.tsa.seasonal_decompose(array_ar, model='additive',period=4) #additive or multiplicative is data specific
grouped_data['seasons']=decomposition.seasonal
trends = decomposition.trend
trend_series = pd.Series(trends)
trend_series.ffill(inplace=True)
trend_series.bfill(inplace=True)
trend_series.interpolate(method='linear', inplace=True)
filled_trend = trend_series.to_numpy()
grouped_data['trend_decomposed']=filled_trend
residuals = decomposition.resid
residuals_series = pd.Series(residuals)
residuals_series.ffill(inplace=True)
residuals_series.bfill(inplace=True)
residuals_series.interpolate(method='linear', inplace=True)
filled_residuals=residuals_series.to_numpy()
grouped_data['residuals_decomposed']=filled_residuals

model = Prophet()

models = [(ARIMASARIMAXModel,{'seasonal':True, 'm':7, 'trace':True}),(ProphetModel, {}),(ARIMASARIMAXModel_,{'seasonal':True, 'm':7, 'trace':True})]

fit_args_map = {
    ARIMASARIMAXModel: {'data': grouped_data, 'target_column': 'seasons', 'future_periods': 8, 'exogenous_columns': None},
    ProphetModel: {'data': grouped_data, 'date_column': 'new_date', 'target_column': 'trend_decomposed'},
    ARIMASARIMAXModel_: {'data': grouped_data, 'target_column': 'residuals_decomposed', 'future_periods': 8, 'exogenous_columns': None}
}

for model_class, model_kwargs in models:
    model = model_class(**model_kwargs)

    fit_args = fit_args_map.get(model_class, {})

    model.fit(**fit_args)
    print(f" Model: {model_class.__name__}")
    print()
    if isinstance(model, ARIMASARIMAXModel):
        predict_arima = model.forecast()
    if isinstance(model, ProphetModel):
        predict_prophet = np.array(model.predict(periods=8, freq='MS'))
    if isinstance(model, ARIMASARIMAXModel_):
        predict_arima_residuals = model.forecast()

residuals=predict_arima_residuals[0]
predict_arima_residuals = residuals.to_numpy()
predict_arima_residuals=5*(predict_arima_residuals)
result_array = 3* (np.array(predict_prophet))
series_data=predict_arima[0]
np_predict_arima = series_data.to_numpy()
np_predict_arima=2*((np_predict_arima))
combined_result=((np.add(result_array,np_predict_arima,predict_arima_residuals))**2)/5

print(combined_result)


last_date = grouped_data['new_date'].max()
new_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=8, freq='MS')
all_dates = pd.concat([grouped_data['new_date'], pd.Series(new_dates)])
all_dates = pd.to_datetime(all_dates).sort_values().reset_index(drop=True)
extended_dates_array = all_dates.to_numpy()

plt.plot(grouped_data['new_date'], grouped_data['count_orignal'], label='Original Data', color='black')
plt.plot(grouped_data_1['new_date'], grouped_data_1['count'], label='Original Data(test)', color='purple')
plt.plot(extended_dates_array, combined_result, color='red', label='combined Predictions')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Original Data and Prophet Model Predictions')
plt.legend()
plt.grid(True)
plt.show()

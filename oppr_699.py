import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from transform import TimeSeriesDifferencer, SquareRootTransformer, SimpleMovingAverage, EWMAOptimizer, \
    BoxCoxTransformer, CustomStandardScaler, ARIMATransformer, RawData
from model import ARIMASARIMAXModel, ProphetModel, ARIMASARIMAXModel_
from statsmodels.tsa.stattools import adfuller, kpss
from prophet import Prophet
import statsmodels.api as sm


data = pd.read_csv('oppr_report_df_989699.csv')
filtered_data = data[~data['account_id'].str.contains(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', na=False)]
data['created_date_local_oppr'] = pd.to_datetime(data['created_date_local_oppr'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
data['close_date_local'] = pd.to_datetime(data['close_date_local'], errors='coerce')
dataset = data[['oppr_id', 'created_date_local_oppr']].copy()
dataset['date'] = pd.to_datetime(dataset['created_date_local_oppr'])
dataset.drop('created_date_local_oppr', axis=1, inplace=True)
dataset['day'] = 1
dataset['month'] = dataset['date'].dt.month
dataset['year'] = dataset['date'].dt.year
dataset['new_date'] = pd.to_datetime(dataset[['year', 'month', 'day']])
grouped_data = dataset.groupby('new_date').agg({'oppr_id': lambda x: ', '.join(x), 'day': 'count'}).rename(
    columns={'day': 'count'}).reset_index()
grouped_data = grouped_data.drop('oppr_id', axis=1)
grouped_data = grouped_data.drop(grouped_data.index[-1])
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
grouped_data_1 = grouped_data[73:]
grouped_data = grouped_data[21:73]
grouped_data['count_orignal']=grouped_data['count']
array_ar=np.array(grouped_data['count'])
decomposition = sm.tsa.seasonal_decompose(array_ar, model='additive',period=4) #additive or multiplicative is data specific
grouped_data['seasons']=decomposition.seasonal
grouped_data
trends = decomposition.trend
trend_series = pd.Series(trends)
trend_series.ffill(inplace=True)
trend_series.bfill(inplace=True)
trend_series.interpolate(method='linear', inplace=True)
filled_trend = trend_series.to_numpy()
grouped_data['trend_decomposed']=np.sqrt(filled_trend)
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
    ARIMASARIMAXModel: {'data': grouped_data, 'target_column': 'seasons', 'future_periods': 5, 'exogenous_columns': None},
    ProphetModel: {'data': grouped_data, 'date_column': 'new_date', 'target_column': 'trend_decomposed'},
    ARIMASARIMAXModel_: {'data': grouped_data, 'target_column': 'residuals_decomposed', 'future_periods': 5, 'exogenous_columns': None}
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
        predict_prophet = np.array(model.predict(periods=5, freq='MS'))
    if isinstance(model, ARIMASARIMAXModel_):
        predict_arima_residuals = model.forecast()

residuals=predict_arima_residuals[0]
predict_arima_residuals = residuals.to_numpy()
predict_arima_residuals=5*(predict_arima_residuals)
result_array = 3* ((np.array(predict_prophet))**2)
series_data=predict_arima[0]
np_predict_arima = series_data.to_numpy()
np_predict_arima=1*((np_predict_arima))
combined_result=((np.add(result_array,np_predict_arima,predict_arima_residuals)))/3
print(combined_result)


last_date = grouped_data['new_date'].max()
new_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=5, freq='MS')
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

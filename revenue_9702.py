import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from transform import TimeSeriesDifferencer, SquareRootTransformer, SimpleMovingAverage, EWMAOptimizer, \
    BoxCoxTransformer, CustomStandardScaler, ARIMATransformer, RawData
from model import ARIMASARIMAXModel, ProphetModel, ARIMASARIMAXModel_, ProphetModel_, ARIMASARIMAXModel__, \
    ARIMASARIMAXModel___
from statsmodels.tsa.stattools import adfuller, kpss
from prophet import Prophet
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.filters.hp_filter import hpfilter

data = pd.read_csv('oppr_report_df_989702.csv', encoding='ISO-8859-1')
filtered_data = data[data['stage_name_cat'] == 'Closed Won']
data['created_date_local_oppr'] = pd.to_datetime(data['created_date_local_oppr'], format='%Y-%m-%d %H:%M:%S', errors='coerce').dt.date
data['close_date_local'] = pd.to_datetime(data['close_date_local'], errors='coerce')
data.loc[data['close_date_local'].dt.year > 2024, 'close_date_local'] = data['created_date_local_oppr']
dataset = data[['amount', 'close_date_local']].copy()
dataset['date'] = dataset['close_date_local'].dt.date
dataset.drop('close_date_local', axis=1, inplace=True)
dataset['date'] = pd.to_datetime(dataset['date'])
dataset['day'] = 1
dataset['month'] = dataset['date'].dt.month
dataset['year'] = dataset['date'].dt.year
dataset['new_date'] = pd.to_datetime(dataset[['year', 'month', 'day']])
grouped_data = dataset.groupby('new_date', as_index=False)['amount'].sum()
grouped_data = grouped_data.drop(grouped_data.index[-1])
grouped_data = grouped_data.drop_duplicates(subset='new_date')
grouped_data = grouped_data.set_index('new_date')
full_range = pd.date_range(start=grouped_data.index.min(), end=grouped_data.index.max(), freq='MS')
grouped_data = grouped_data.reindex(full_range, fill_value=0)
grouped_data = grouped_data.reset_index()
grouped_data.rename(columns={'index': 'new_date'}, inplace=True)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
grouped_data_1=grouped_data[76:]
grouped_data=grouped_data[42:77]
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
grouped_data['count_orignal']=grouped_data['amount']
array_ar=np.array(grouped_data['amount'])
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
grouped_data['amount']=np.sqrt(np.array(grouped_data['amount']))
models = [(ARIMASARIMAXModel__,{'seasonal':True, 'm':7, 'trace':True}),(ProphetModel_, {}),(ARIMASARIMAXModel___,{'seasonal':True, 'm':7, 'trace':True})]
fit_args_map = {
    ARIMASARIMAXModel__: {'data': grouped_data, 'target_column': 'amount', 'future_periods': 5, 'exogenous_columns': None},
    ProphetModel_: {'data': grouped_data, 'date_column': 'new_date', 'target_column': 'trend_decomposed'},
    ARIMASARIMAXModel___: {'data': grouped_data, 'target_column': 'trend_decomposed', 'future_periods': 5, 'exogenous_columns': None}
}

for model_class, model_kwargs in models:
    model = model_class(**model_kwargs)

    fit_args = fit_args_map.get(model_class, {})

    model.fit(**fit_args)
    print(f" Model: {model_class.__name__}")
    print()
    if isinstance(model, ARIMASARIMAXModel__):
        predict_arima = model.forecast()
    if isinstance(model, ProphetModel_):
        predict_prophet_ = np.array(model.predict(periods=5, freq='MS'))
    if isinstance(model, ARIMASARIMAXModel___):
        predict_arima_trend = model.forecast()

# print(predict_prophet)
trends_arima=predict_arima_trend[0]
predict_arima_trends = trends_arima.to_numpy()
result_array = (np.array(predict_prophet_))
series_data=predict_arima[0]
arima_amount = ((series_data.to_numpy())**2)
net_trends=np.add(predict_arima_trends,result_array)
net_trends=(np.add(predict_arima_trends,result_array))/2
combined_result=(np.add(net_trends,arima_amount))/1
print(combined_result)


last_date = grouped_data['new_date'].max()
new_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=5, freq='MS')
all_dates = pd.concat([grouped_data['new_date'], pd.Series(new_dates)])
all_dates = pd.to_datetime(all_dates).sort_values().reset_index(drop=True)
extended_dates_array = all_dates.to_numpy()

plt.plot(grouped_data['new_date'], grouped_data['count_orignal'], label='Original Data', color='black')
plt.plot(grouped_data_1['new_date'], grouped_data_1['amount'], label='Original Data(test)', color='purple')
plt.plot(extended_dates_array, combined_result, color='red', label='combined Predictions')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Original Data and Prophet Model Predictions')
plt.legend()
plt.grid(True)
plt.show()
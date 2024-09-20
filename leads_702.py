import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from model_leads import ARIMASARIMAXModel, ProphetModel
from prophet import Prophet
import statsmodels.api as sm


data = pd.read_csv('leads_report_df_989702.csv')
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
grouped_data_1 = grouped_data[19:]
grouped_data_pycaret = grouped_data.copy()
grouped_data = grouped_data[3:19]
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
first_valid_index = trend_series.first_valid_index()
filled_trend = trend_series.to_numpy()
grouped_data['trend_decomposed']=filled_trend
grouped_data_season=grouped_data.copy()
grouped_data_season['new_date'] = pd.to_datetime(grouped_data_season['new_date'])
grouped_data_season.set_index('new_date', inplace=True)
grouped_data_season.index = pd.to_datetime(grouped_data_season.index)
grouped_data_season = grouped_data_season.asfreq('MS')  # Adjust 'MS' to match your data frequency

order = (1, 1, 1)  # (p, d, q)
seasonal_order = (1, 1, 1, 12)  # (P, D, Q, m), e.g., m=12 for monthly seasonality
# Fit the SARIMA model
model = SARIMAX(grouped_data_season['seasons'],
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False)
results = model.fit(disp=False)
forecast_steps = 4 # Specify the number of steps to forecast
forecast = results.get_forecast(steps=forecast_steps)
in_sample_pred = results.get_prediction(start=0, end=len(grouped_data_season) - 1)
in_sample_pred_mean = in_sample_pred.predicted_mean
in_sample_conf_int = in_sample_pred.conf_int()

in_sample_df = pd.DataFrame({
    'Actual': grouped_data_season['seasons'],
    'Predicted': in_sample_pred_mean,
    'Lower CI': in_sample_conf_int.iloc[:, 0],
    'Upper CI': in_sample_conf_int.iloc[:, 1]
})
forecast_steps = 4  # Specify the number of steps to forecast
forecast = results.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()
forecast_df = pd.DataFrame({
    'Forecast': forecast_mean,
    'Lower CI': forecast_conf_int.iloc[:, 0],
    'Upper CI': forecast_conf_int.iloc[:, 1]
})
in_sample_actual = grouped_data_season['seasons']
in_sample_predicted = in_sample_pred_mean
in_sample_lower_ci = in_sample_conf_int.iloc[:, 0]
in_sample_upper_ci = in_sample_conf_int.iloc[:, 1]
in_sample_df = pd.DataFrame({
    'Actual': in_sample_actual,
    'Predicted': in_sample_predicted,
    'Lower CI': in_sample_lower_ci,
    'Upper CI': in_sample_upper_ci
})
future_dates = pd.date_range(start=grouped_data_season.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
future_forecast_df = pd.DataFrame(index=future_dates)
future_forecast_df['Forecast'] = forecast_mean
future_forecast_df['Lower CI'] = forecast_conf_int.iloc[:, 0]
future_forecast_df['Upper CI'] = forecast_conf_int.iloc[:, 1]
combined_df = pd.concat([in_sample_df, future_forecast_df])
df = pd.DataFrame(combined_df, index=pd.date_range(start='2022-10-01', periods=20, freq='MS'))
predicted_values = df['Predicted'].iloc[:16].to_numpy()
forecast_values = df['Forecast'].iloc[-4:].to_numpy()
combined_array = np.concatenate([predicted_values, forecast_values])
combined_array_seasons= combined_array[~np.isnan(combined_array)]


model = Prophet()

models = [(ARIMASARIMAXModel,{'seasonal':True, 'm':7, 'trace':True}),(ProphetModel, {})]

fit_args_map = {
    ARIMASARIMAXModel: {'data': grouped_data, 'target_column': 'seasons', 'future_periods': 4, 'exogenous_columns': None},
    ProphetModel: {'data': grouped_data, 'date_column': 'new_date', 'target_column': 'trend_decomposed'},

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
        predict_prophet = np.array(model.predict(periods=4, freq='MS'))

result_array = 5* np.array(predict_prophet)
np_predict_arima=15*(combined_array_seasons)
combined_result=(np.add(result_array,np_predict_arima))**2/(5**2)

last_date = grouped_data['new_date'].max()
new_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=4, freq='MS')
all_dates = pd.concat([grouped_data['new_date'], pd.Series(new_dates)])
all_dates = pd.to_datetime(all_dates).sort_values().reset_index(drop=True)
extended_dates_array = all_dates.to_numpy()

plt.plot(grouped_data['new_date'], grouped_data['count_orignal'], label='Original Data', color='black')
plt.plot(grouped_data_1['new_date'], grouped_data_1['count'], label='Original Data(test)', color='purple')
# plt.plot(extended_dates_array, result_array, color='green', label='Prophet Predictions')
# plt.plot(extended_dates_array, np_predict_arima, color='orange', label='Arima Predictions')
plt.plot(extended_dates_array, combined_result, color='red', label='combined Predictions')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Original Data and Prophet Model Predictions')
plt.legend()
plt.grid(True)
plt.show()

# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# from pmdarima import auto_arima
# from prophet import Prophet
# import statsmodels.api as sm
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from transform import TimeSeriesDifferencer,SquareRootTransformer,SimpleMovingAverage,EWMAOptimizer,BoxCoxTransformer,CustomStandardScaler,ARIMATransformer,RawData
# from model import ARIMASARIMAXModel,ProphetModel
# from nixtla import NixtlaClient
# # from pycaret.regression import *
# from statsmodels.tsa.arima.model import ARIMA
# # from pycarret_model import pycaret_forecasting
# from sklearn.metrics import mean_squared_error
# from statsmodels.tsa.stattools import adfuller, kpss
#
# modelpackages={'pmdarima':'pmdarima'}
# # for package in modelpackages:
# #     try:
# #         import package
# #     except:
# #         checkForPackages(modelpackages[package])
#
#
#
#
# # Load and preprocess the data
# data = pd.read_csv('leads_report_df_989699.csv')
# dataset = data[['email', 'created_at']].copy()
# dataset['date'] = dataset['created_at'].str.split(' ').str[0]
# dataset.drop('created_at', axis=1, inplace=True)
# dataset['date'] = pd.to_datetime(dataset['date'])
# dataset['day'] = 1
# dataset['month'] = dataset['date'].dt.month
# dataset['year'] = dataset['date'].dt.year
# dataset['new_date'] = pd.to_datetime(dataset[['year', 'month', 'day']])
# grouped_data = dataset.groupby('new_date').agg({'email': lambda x: ', '.join(x), 'day': 'count'}).rename(columns={'day': 'count'}).reset_index()
# grouped_data = grouped_data.drop('email', axis=1)
# grouped_data = grouped_data.drop(grouped_data.index[-1])
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# print(grouped_data)
# pd.reset_option('display.max_rows')
# pd.reset_option('display.max_columns')
# grouped_data_1=grouped_data[167:173]
# grouped_data_pycaret=grouped_data.copy()
# grouped_data=grouped_data[140:167]
#
#
# def adf_test(timeseries):
#     print("Results of Dickey-Fuller Test:")
#     dftest = adfuller(timeseries, autolag="AIC")
#     dfoutput = pd.Series(
#         dftest[0:4],
#         index=[
#             "Test Statistic",
#             "p-value",
#             "#Lags Used",
#             "Number of Observations Used",
#         ],
#     )
#     for key, value in dftest[4].items():
#         dfoutput["Critical Value (%s)" % key] = value
#     print(dfoutput)
# def kpss_test(timeseries):
#     print("Results of KPSS Test:")
#     kpsstest = kpss(timeseries, regression="c", nlags="auto")
#     kpss_output = pd.Series(
#         kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
#     )
#     for key, value in kpsstest[3].items():
#         kpss_output["Critical Value (%s)" % key] = value
#     print(kpss_output)
#
# adf_test(grouped_data["count"])
#
# kpss_test(grouped_data["count"])
#
#
#
#
# transformations = [
#     # (TimeSeriesDifferencer, {'k_diff': 1, 'fill_na_value': None}),
#     (SquareRootTransformer, {'shift_negative': True, 'shift_value': None}),
#     # (SimpleMovingAverage, {'window_size': 5}),
#     # (EWMAOptimizer, {'alpha_values': np.linspace(0.01, 1, 100)}),
#     # (BoxCoxTransformer, {'lmbda_range': (-5, 5), 'num_steps': 1000}),
#     # (CustomStandardScaler, {}),
#     # (ARIMATransformer,{'p':20, 'd':0, 'q':20}),
#     # (RawData, {})
#     ]
# # custom_seasonalities = [
# #     {'name': 'Q1', 'period': 7, 'fourier_order': 5},  # Q1 seasonality with 89 days
# #     {'name': 'Q2', 'period': 8, 'fourier_order': 5},  # Q2 seasonality with 92 days
# #     {'name': 'Q3', 'period': 9, 'fourier_order': 5},  # Q3 seasonality with 93 days
# #
# # ]
# models=[(ARIMASARIMAXModel,{'seasonal':True, 'm':4, 'trace':True}),(ProphetModel,{})]
#
# transform_args_map = {
#     # TimeSeriesDifferencer: {'data': grouped_data, 'column': 'count'},
#     SquareRootTransformer: {'data': grouped_data, 'column': 'count'},
#     # SimpleMovingAverage: {'data': grouped_data, 'column_name': 'count'},
#     # EWMAOptimizer: {'data': grouped_data,'test_size':0.2},
#     # BoxCoxTransformer: {'data': grouped_data},
#     # CustomStandardScaler: {'data': grouped_data},
#     # ARIMATransformer:{'data': grouped_data,'column': 'count'},
#     # RawData: {'data': grouped_data}
# }
#
# fit_args_map = {
#     ARIMASARIMAXModel: {'data': grouped_data, 'target_column': 'count', 'future_periods':6,'exogenous_columns': None},
#     ProphetModel: {'data': grouped_data, 'date_column': 'new_date', 'target_column': 'count'}
# }
# future_periods = 6
#
# flag=0;
# # Iterate over each transformation and model combination
# for transform_class, transform_kwargs in transformations:
#     # Get default arguments for the transformation from the map
#     if (transform_class == SquareRootTransformer):
#         flag=1
#     elif (transform_class==RawData):
#         flag=2
#     else:
#         flag=0
#     if (transform_class==ARIMATransformer):
#         best_p, best_d, best_q = None, None, None
#         min_mse = float('inf')
#         arima_transformer = ARIMATransformer(p=14, d=2, q=8)
#         arima_transformer.fit(grouped_data, column='count')
#         transformed_data = arima_transformer.transform(grouped_data, column='count')
#         predictions = transformed_data[-6:]
#
#
#     else:
#         transform_default_args = dict(transform_args_map.get(transform_class, {}))
#         transform_default_args.update(transform_kwargs)  # Update with specific kwargs if provided
#
#         transform = transform_class(**transform_default_args)
#
#         if hasattr(transform, 'fit_transform'):
#             if 'data' in transform_default_args and 'column' in transform_default_args:
#                 transformed_data = transform.fit_transform(grouped_data, transform_default_args['column'])
#             elif 'column_name' in transform_default_args:
#                 transformed_data = transform.fit_transform(grouped_data, transform_default_args['column_name'])
#             else:
#                 transformed_data = transform.fit_transform(grouped_data)  # Adjust based on implementation
#         else:
#             transform.fit(grouped_data, 'count')
#             transformed_data = transform.transform(grouped_data)
#
#     for model_class, model_kwargs in models:
#         model = model_class(**model_kwargs)
#
#         fit_args = fit_args_map.get(model_class, {})
#
#         fit_args.update({'data': transformed_data})  # Add transformed_data if required
#
#         model.fit(**fit_args)
#         print(f"Transformation: {transform_class.__name__}, Model: {model_class.__name__}")
#         print()
#
#         if isinstance(model, ARIMASARIMAXModel):
#             if flag==1:
#                 predict_arima=model.forecast()
#             elif flag==0:
#                 predict_arima_1=model.forecast()
#             else:
#                 forecast, conf_int = model.forecast()
#             # model.plot_forecast()
#             # model.plot_acf_pacf()
#         elif isinstance(model, ProphetModel):
#             if flag==2:
#                 predict_prophet=np.array(model.predict(periods=6, freq='MS'))
#             elif flag==1:
#                 predict_prophet_sqrt=np.array(model.predict(periods=6, freq='MS'))
#             else:
#                 model.predict(periods=6, freq='MS')
#             # model.plot(grouped_data, date_column='new_date', target_column='count')
#
# print("prophet prediction",predict_prophet_sqrt)
# print (len(predict_prophet_sqrt))
# series_data=predict_arima[0]
# np_predict_arima = series_data.to_numpy()
# squared_array_arima = np_predict_arima ** 2
# print(squared_array_arima)
# print(len(predict_prophet_sqrt))
# squared_array_prophet=predict_prophet_sqrt**2
# print(len(np_predict_arima))
# import pdb;pdb.set_trace()
# result_array = (2*squared_array_prophet + squared_array_arima)/3
# print("resultanat array",result_array)
#
# last_date = grouped_data['new_date'].max()
# new_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='MS')
# all_dates = pd.concat([grouped_data['new_date'], pd.Series(new_dates)])
# all_dates = pd.to_datetime(all_dates).sort_values().reset_index(drop=True)
# extended_dates_array = all_dates.to_numpy()
#
# # plt.plot(grouped_data['new_date'], grouped_data['count'], label='Original Data', color='black')
# # plt.plot(grouped_data_1['new_date'], grouped_data_1['count'], label='Original Data(test)', color='purple')
# # plt.plot(extended_dates_array, squared_array, color='green', label='ARIMA Predictions')
# # plt.xlabel('Date')
# # plt.ylabel('Count')
# # plt.title('Original Data and ARIMA Model Predictions')
# # plt.legend()
# # plt.grid(True)
# # plt.show()
#
# # Plotting the combined ARIMA+Prophet model predictions
# plt.plot(grouped_data['new_date'], grouped_data['count'], label='Original Data', color='black')
# plt.plot(grouped_data_1['new_date'], grouped_data_1['count'], label='Original Data(test)', color='purple')
# plt.plot(extended_dates_array, result_array, color='green', label='ARIMA+Prophet Predictions')
# plt.xlabel('Date')
# plt.ylabel('Count')
# plt.title('Original Data and ARIMA+Prophet Model Predictions')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # Debugging the date array
# print("dates",extended_dates_array)
#
#
#
#
# # pycaret_forecasting(grouped_data_pycaret)
# # import pdb;pdb.set_trace()
# # import pdb;pdb.set_trace()
# # nixtla_client = NixtlaClient(
# #     api_key = 'nixtla-tok-S1O8WolpN5w3P9zxoZfxPA4IhTvJvGAF5s1BezjTTuQi3yfN9fnTIyl2wAljmzW1J5Iw3Dv5tTJcjJZS'
# # )
# # nixtla_client.validate_api_key()
# # import pdb;pdb.set_trace()
# #
# # nixtla_client.plot(grouped_data, time_col='new_date', target_col='count')
# # timegpt_fcst_df = nixtla_client.forecast(df=grouped_data, h=12, freq='MS', time_col='new_date', target_col='count')
# # timegpt_fcst_df.head()
# # nixtla_client.plot(grouped_data, timegpt_fcst_df, time_col='new_date', target_col='count')
#
#
#
# # test_3 code
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# from pmdarima import auto_arima
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from transform import TimeSeriesDifferencer, SquareRootTransformer, SimpleMovingAverage, EWMAOptimizer, \
#     BoxCoxTransformer, CustomStandardScaler, ARIMATransformer, RawData
# from model import ARIMASARIMAXModel, ProphetModel
# from statsmodels.tsa.stattools import adfuller, kpss
# from prophet import Prophet
# import statsmodels.api as sm
# from statsmodels.tsa.filters.hp_filter import hpfilter
#
# data = pd.read_csv('leads_report_df_989702.csv')
# dataset = data[['email', 'created_at']].copy()
# dataset['date'] = dataset['created_at'].str.split(' ').str[0]
# dataset.drop('created_at', axis=1, inplace=True)
# dataset['date'] = pd.to_datetime(dataset['date'])
# dataset['day'] = 1
# dataset['month'] = dataset['date'].dt.month
# dataset['year'] = dataset['date'].dt.year
# dataset['new_date'] = pd.to_datetime(dataset[['year', 'month', 'day']])
# grouped_data = dataset.groupby('new_date').agg({'email': lambda x: ', '.join(x), 'day': 'count'}).rename(
#     columns={'day': 'count'}).reset_index()
# grouped_data = grouped_data.drop('email', axis=1)
# grouped_data = grouped_data.drop(grouped_data.index[-1])
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.reset_option('display.max_rows')
# pd.reset_option('display.max_columns')
# grouped_data_1 = grouped_data[19:]
# grouped_data_pycaret = grouped_data.copy()
# grouped_data = grouped_data[3:19]
# grouped_data['count_orignal']=grouped_data['count']
# count_sqrt=np.array(grouped_data['count'])
# sqrt_transformed = np.sqrt(count_sqrt)
# grouped_data['count']=sqrt_transformed
# print(grouped_data.columns)
# print(grouped_data)
# # grouped_data['count']=grouped_data['count'].clip(upper=8000)
#
# # gdp_cycle, gdp_trend = hpfilter(grouped_data['count'], lamb=14400)
# # gdp_segr = grouped_data[['count']]
# # gdp_segr['cycle'] = gdp_cycle
# # gdp_segr['trend'] = gdp_trend
# # print(gdp_segr)
# #
# # gdp_cycle.plot()
# array_ar=np.array(grouped_data['count'])
# decomposition = sm.tsa.seasonal_decompose(array_ar, model='additive',period=4) #additive or multiplicative is data specific
# print('trend',decomposition.trend)
# print('season',decomposition.seasonal)
# print('residuals',decomposition.resid)
#
# fig = decomposition.plot()
# plt.show()
# grouped_data['seasons']=decomposition.seasonal
# trends = decomposition.trend
# trend_series = pd.Series(trends)
# trend_series.ffill(inplace=True)
# trend_series.bfill(inplace=True)
# trend_series.interpolate(method='linear', inplace=True)
# # Fill NaNs at the start
# first_valid_index = trend_series.first_valid_index()
# filled_trend = trend_series.to_numpy()
# grouped_data['trend_decomposed']=filled_trend
# print(grouped_data)
#
#
# # pdb.set_trace()
#
# grouped_data_season=grouped_data.copy()
# grouped_data_season['new_date'] = pd.to_datetime(grouped_data_season['new_date'])
# grouped_data_season.set_index('new_date', inplace=True)
# grouped_data_season.index = pd.to_datetime(grouped_data_season.index)
# grouped_data_season = grouped_data_season.asfreq('MS')  # Adjust 'MS' to match your data frequency
#
# # Specify your SARIMA model parameters (replace with your chosen values)
# order = (1, 1, 1)  # (p, d, q)
# seasonal_order = (1, 1, 1, 12)  # (P, D, Q, m), e.g., m=12 for monthly seasonality
# # Fit the SARIMA model
# model = SARIMAX(grouped_data_season['seasons'],
#                 order=order,
#                 seasonal_order=seasonal_order,
#                 enforce_stationarity=False,
#                 enforce_invertibility=False)
# # Fit the model
# results = model.fit(disp=False)
# # Make predictions (in-sample and future)
# forecast_steps = 4 # Specify the number of steps to forecast
# forecast = results.get_forecast(steps=forecast_steps)
# # print(forecast['trend','yhat','season'])
# in_sample_pred = results.get_prediction(start=0, end=len(grouped_data_season) - 1)
# in_sample_pred_mean = in_sample_pred.predicted_mean
# in_sample_conf_int = in_sample_pred.conf_int()
#
# # Print historical (in-sample) predictions
# print("In-Sample (Historical) Predictions:")
# in_sample_df = pd.DataFrame({
#     'Actual': grouped_data_season['seasons'],
#     'Predicted': in_sample_pred_mean,
#     'Lower CI': in_sample_conf_int.iloc[:, 0],
#     'Upper CI': in_sample_conf_int.iloc[:, 1]
# })
# print(in_sample_df)
#
# # Make future (out-of-sample) predictions
# forecast_steps = 4  # Specify the number of steps to forecast
# forecast = results.get_forecast(steps=forecast_steps)
#
# # Get forecasted values and confidence intervals
# forecast_mean = forecast.predicted_mean
# forecast_conf_int = forecast.conf_int()
#
# # Print future (out-of-sample) predictions
# print("\nFuture (Out-of-Sample) Predictions:")
# forecast_df = pd.DataFrame({
#     'Forecast': forecast_mean,
#     'Lower CI': forecast_conf_int.iloc[:, 0],
#     'Upper CI': forecast_conf_int.iloc[:, 1]
# })
# in_sample_actual = grouped_data_season['seasons']
# in_sample_predicted = in_sample_pred_mean
# in_sample_lower_ci = in_sample_conf_int.iloc[:, 0]
# in_sample_upper_ci = in_sample_conf_int.iloc[:, 1]
#
# # Create a DataFrame to handle the in-sample data
# in_sample_df = pd.DataFrame({
#     'Actual': in_sample_actual,
#     'Predicted': in_sample_predicted,
#     'Lower CI': in_sample_lower_ci,
#     'Upper CI': in_sample_upper_ci
# })
#
# # Out-of-Sample Predictions
# future_dates = pd.date_range(start=grouped_data_season.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
# future_forecast_df = pd.DataFrame(index=future_dates)
# future_forecast_df['Forecast'] = forecast_mean
# future_forecast_df['Lower CI'] = forecast_conf_int.iloc[:, 0]
# future_forecast_df['Upper CI'] = forecast_conf_int.iloc[:, 1]
#
# # Combine in-sample and out-of-sample data
# combined_df = pd.concat([in_sample_df, future_forecast_df])
# df = pd.DataFrame(combined_df, index=pd.date_range(start='2022-10-01', periods=20, freq='MS'))
# predicted_values = df['Predicted'].iloc[:16].to_numpy()
# forecast_values = df['Forecast'].iloc[-4:].to_numpy()
#
# # Combine the two arrays
# combined_array = np.concatenate([predicted_values, forecast_values])
#
# # Remove NaN values if any
# combined_array_seasons= combined_array[~np.isnan(combined_array)]
#
# import pdb;pdb.set_trace()
#
# model = Prophet()
#
# models = [(ARIMASARIMAXModel,{'seasonal':True, 'm':7, 'trace':True}),(ProphetModel, {})]
#
# fit_args_map = {
#     ARIMASARIMAXModel: {'data': grouped_data, 'target_column': 'seasons', 'future_periods': 4, 'exogenous_columns': None},
#     ProphetModel: {'data': grouped_data, 'date_column': 'new_date', 'target_column': 'trend_decomposed'},
#
# }
# for model_class, model_kwargs in models:
#     model = model_class(**model_kwargs)
#
#     fit_args = fit_args_map.get(model_class, {})
#
#     model.fit(**fit_args)
#     print(f" Model: {model_class.__name__}")
#     print()
#     if isinstance(model, ARIMASARIMAXModel):
#         predict_arima = model.forecast()
#     if isinstance(model, ProphetModel):
#         predict_prophet = np.array(model.predict(periods=4, freq='MS'))
#
# print('prophet_1',predict_prophet)
# result_array = 5* np.array(predict_prophet)
# print('seasons',combined_array_seasons)
#
#
# # print('seasons',predict_arima)
# # series_data=predict_arima[0]
# # np_predict_arima = series_data.to_numpy()
# # print('arima',np_predict_arima)
#
# np_predict_arima=15*(combined_array_seasons)
# combined_result=(np.add(result_array,np_predict_arima))**2/(5**2)
#
#
#
# last_date = grouped_data['new_date'].max()
# new_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=4, freq='MS')
# all_dates = pd.concat([grouped_data['new_date'], pd.Series(new_dates)])
# all_dates = pd.to_datetime(all_dates).sort_values().reset_index(drop=True)
# extended_dates_array = all_dates.to_numpy()
#
# plt.plot(grouped_data['new_date'], grouped_data['count_orignal'], label='Original Data', color='black')
# plt.plot(grouped_data_1['new_date'], grouped_data_1['count'], label='Original Data(test)', color='purple')
# # plt.plot(extended_dates_array, result_array, color='green', label='Prophet Predictions')
# # plt.plot(extended_dates_array, np_predict_arima, color='orange', label='Arima Predictions')
# plt.plot(extended_dates_array, combined_result, color='red', label='combined Predictions')
# plt.xlabel('Date')
# plt.ylabel('Count')
# plt.title('Original Data and Prophet Model Predictions')
# plt.legend()
# plt.grid(True)
# plt.show()
#
#
# # test-2
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# from transform import TimeSeriesDifferencer, SquareRootTransformer, SimpleMovingAverage, EWMAOptimizer, \
#     BoxCoxTransformer, CustomStandardScaler, ARIMATransformer, RawData
# from model import ARIMASARIMAXModel, ProphetModel, ARIMASARIMAXModel_
# from statsmodels.tsa.stattools import adfuller, kpss
# from prophet import Prophet
# import statsmodels.api as sm
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.filters.hp_filter import hpfilter
#
# data = pd.read_csv('leads_report_df_989699.csv')
# dataset = data[['email', 'created_at']].copy()
# dataset['date'] = dataset['created_at'].str.split(' ').str[0]
# dataset.drop('created_at', axis=1, inplace=True)
# dataset['date'] = pd.to_datetime(dataset['date'])
# dataset['day'] = 1
# dataset['month'] = dataset['date'].dt.month
# dataset['year'] = dataset['date'].dt.year
# dataset['new_date'] = pd.to_datetime(dataset[['year', 'month', 'day']])
# grouped_data = dataset.groupby('new_date').agg({'email': lambda x: ', '.join(x), 'day': 'count'}).rename(
#     columns={'day': 'count'}).reset_index()
# grouped_data = grouped_data.drop('email', axis=1)
# grouped_data = grouped_data.drop(grouped_data.index[-1])
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# print(grouped_data)
# pd.reset_option('display.max_rows')
# pd.reset_option('display.max_columns')
#
# grouped_data_1 = grouped_data[168:173]
# grouped_data_pycaret = grouped_data.copy()
# grouped_data = grouped_data[141:169]
# grouped_data['count_orignal']=grouped_data['count']
# count_sqrt=np.array(grouped_data['count'])
# sqrt_transformed = np.sqrt(count_sqrt)
# grouped_data['count']=sqrt_transformed
# print(grouped_data.columns)
# print(grouped_data)
# # grouped_data['count']=grouped_data['count'].clip(upper=8000)
#
# # gdp_cycle, gdp_trend = hpfilter(grouped_data['count'], lamb=14400)
# # gdp_segr = grouped_data[['count']]
# # gdp_segr['cycle'] = gdp_cycle
# # gdp_segr['trend'] = gdp_trend
# # print(gdp_segr)
# #
# # gdp_cycle.plot()
# array_ar=np.array(grouped_data['count'])
# decomposition = sm.tsa.seasonal_decompose(array_ar, model='additive',period=4) #additive or multiplicative is data specific
# print('trend',decomposition.trend)
# print('season',decomposition.seasonal)
# print('residuals',decomposition.resid)
#
# fig = decomposition.plot()
# plt.show()
# grouped_data['seasons']=decomposition.seasonal
# trends = decomposition.trend
# trend_series = pd.Series(trends)
# trend_series.ffill(inplace=True)
# trend_series.bfill(inplace=True)
# trend_series.interpolate(method='linear', inplace=True)
# filled_trend = trend_series.to_numpy()
# grouped_data['trend_decomposed']=filled_trend
# residuals = decomposition.resid
# residuals_series = pd.Series(residuals)
# residuals_series.ffill(inplace=True)
# residuals_series.bfill(inplace=True)
# residuals_series.interpolate(method='linear', inplace=True)
# filled_residuals=residuals_series.to_numpy()
# grouped_data['residuals_decomposed']=filled_residuals
#
# model = Prophet()
#
# models = [(ARIMASARIMAXModel,{'seasonal':True, 'm':7, 'trace':True}),(ProphetModel, {}),(ARIMASARIMAXModel_,{'seasonal':True, 'm':7, 'trace':True})]
#
# fit_args_map = {
#     ARIMASARIMAXModel: {'data': grouped_data, 'target_column': 'seasons', 'future_periods': 8, 'exogenous_columns': None},
#     ProphetModel: {'data': grouped_data, 'date_column': 'new_date', 'target_column': 'trend_decomposed'},
#     ARIMASARIMAXModel_: {'data': grouped_data, 'target_column': 'residuals_decomposed', 'future_periods': 8, 'exogenous_columns': None}
# }
#
# for model_class, model_kwargs in models:
#     model = model_class(**model_kwargs)
#
#     fit_args = fit_args_map.get(model_class, {})
#
#     model.fit(**fit_args)
#     print(f" Model: {model_class.__name__}")
#     print()
#     if isinstance(model, ARIMASARIMAXModel):
#         predict_arima = model.forecast()
#     if isinstance(model, ProphetModel):
#         predict_prophet = np.array(model.predict(periods=8, freq='MS'))
#     if isinstance(model, ARIMASARIMAXModel_):
#         predict_arima_residuals = model.forecast()
#
# print(predict_prophet)
# residuals=predict_arima_residuals[0]
# predict_arima_residuals = residuals.to_numpy()
# predict_arima_residuals=5*(predict_arima_residuals)
# print('residuals',predict_arima_residuals)
# result_array = 3* (np.array(predict_prophet))
# print('prophet',result_array)
# series_data=predict_arima[0]
# np_predict_arima = series_data.to_numpy()
# np_predict_arima=2*((np_predict_arima))
# print('arima',np_predict_arima)
# combined_result=((np.add(result_array,np_predict_arima,predict_arima_residuals))**2)/5
#
# print(combined_result)
#
#
# last_date = grouped_data['new_date'].max()
# new_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=8, freq='MS')
# all_dates = pd.concat([grouped_data['new_date'], pd.Series(new_dates)])
# all_dates = pd.to_datetime(all_dates).sort_values().reset_index(drop=True)
# extended_dates_array = all_dates.to_numpy()
#
# plt.plot(grouped_data['new_date'], grouped_data['count_orignal'], label='Original Data', color='black')
# plt.plot(grouped_data_1['new_date'], grouped_data_1['count'], label='Original Data(test)', color='purple')
# # plt.plot(extended_dates_array, result_array, color='green', label='Prophet Predictions')
# # plt.plot(extended_dates_array, np_predict_arima, color='orange', label='Arima Predictions')
# plt.plot(extended_dates_array, combined_result, color='red', label='combined Predictions')
# plt.xlabel('Date')
# plt.ylabel('Count')
# plt.title('Original Data and Prophet Model Predictions')
# plt.legend()
# plt.grid(True)
# plt.show()
#
#
# # lightgbm_prophet
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from prophet import Prophet
# from lightgbm import LGBMRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
#
#
# data = pd.read_csv('leads_report_df_989699.csv')
# dataset = data[['email', 'created_at']].copy()
# dataset['date'] = dataset['created_at'].str.split(' ').str[0]
# dataset.drop('created_at', axis=1, inplace=True)
# dataset['date'] = pd.to_datetime(dataset['date'])
# dataset['day'] = 1
# dataset['month'] = dataset['date'].dt.month
# dataset['year'] = dataset['date'].dt.year
# dataset['new_date'] = pd.to_datetime(dataset[['year', 'month', 'day']])
# grouped_data = dataset.groupby('new_date').agg({'email': lambda x: ', '.join(x), 'day': 'count'}).rename(columns={'day': 'count'}).reset_index()
# grouped_data = grouped_data.drop('email', axis=1)
# grouped_data = grouped_data.drop(grouped_data.index[-1])
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# print(grouped_data[119:])
# grouped_data=grouped_data[119:]
# grouped_data_1=grouped_data.copy()
# grouped_data_1 = grouped_data_1.reset_index(drop=True)
#
# # plt.plot(grouped_data['new_date'],grouped_data['count'])
# # plt.grid()
# # plt.show()
#
#
# def prophet_features(df, horizon=10):
#     temp_df = df.reset_index()
#     temp_df = temp_df[['new_date', 'count']]
#     temp_df.rename(columns={'new_date': 'ds', 'count': 'y'}, inplace=True)
#     temp_df['cap'] = 9500
#     # import pdb;pdb.set_trace()
#     # Using the data from the previous week as an example for validation
#     train, test = temp_df.iloc[:-horizon, :], temp_df.iloc[-horizon:, :]
#
#     # Define the Prophet model
#     m = Prophet(
#         growth='linear',
#         seasonality_mode='additive'
#     )
#     # m.add_seasonality(name='6months', period=6, fourier_order=5)
#
#     m.fit(train)
#     # Extract features from the data, using Prophet to predict the training set
#     predictions_train = m.predict(train.drop('y', axis=1))
#     # Use Prophet to extract features from the data to predict the test set
#     predictions_test = m.predict(test.drop('y', axis=1))
#     # Combine predictions from the training and test sets
#     predictions = pd.concat([predictions_train, predictions_test], axis=0)
#
#     return predictions
#
# def train_time_series_with_folds_autoreg_prophet_features(df, horizon=10, lags=[1,2]):
#     new_prophet_features = prophet_features(df, horizon=horizon)
#     df.reset_index(inplace=True)
#
#     # Merge the Prophet features dataframe with our initial dataframe
#     df = pd.merge(df, new_prophet_features, left_on=['new_date'], right_on=['ds'], how='inner')
#     df.drop('ds', axis=1, inplace=True)
#     df.set_index('new_date', inplace=True)
#
#     # Use Prophet predictions to create some lag variables (yhat column)
#     for lag in lags:
#         df[f'yhat_lag_{lag}'] = df['yhat'].shift(lag)
#     df.dropna(axis=0, how='any')
#
#     X = df.drop('count', axis=1)
#     y = df['count']
#
#     # Using the data from the previous week as an example for validation
#     X_train, X_test = X.iloc[:-horizon, :], X.iloc[-horizon:, :]
#     y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]
#
#     # Define the LightGBM model, train, and make predictions
#     model = LGBMRegressor(random_state=42)
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#
#     # Calculate MAE
#     mae = np.round(mean_absolute_error(y_test, predictions), 3)
#
#     fig = plt.figure(figsize=(16, 6))
#     plt.title(f'Real vs Prediction - MAE {mae}', fontsize=20)
#     plt.plot(y_test, color='red')
#     plt.plot(pd.Series(predictions, index=y_test.index), color='green')
#     plt.plot(y_train, color='black')
#     plt.xlabel('month', fontsize=16)
#     plt.ylabel('count', fontsize=16)
#     plt.legend(labels=['Real', 'Prediction'], fontsize=16)
#     plt.grid()
#     plt.show()
#
# print(prophet_features(grouped_data))
# # train_time_series_with_folds_autoreg_prophet_features(grouped_data)
#
#
# prophet_features_1=prophet_features(grouped_data)
# prophet_features_1 = prophet_features_1.reset_index(drop=True)
# print(grouped_data_1['count'][0])
#
# for i in range(0,54):
#     # grouped_data_1['count'][i]=(grouped_data_1['count'][i]-(prophet_features_1['trend'][i]))
#     grouped_data_1['count'][i]=(grouped_data_1['count'][i]-(prophet_features_1['yearly'][i]))
#     # grouped_data_1['count'][i]=(grouped_data_1['count'][i]-(prophet_features_1['6months'][i]))
#
# print(grouped_data_1)
#
# plt.plot(grouped_data_1['new_date'],grouped_data_1['count'])
# plt.grid()
# plt.show()
#
#
# # pycaret_model
# from pycaret.regression import setup, compare_models, tune_model, predict_model
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
#
#
# def pycaret_forecasting(data):
#     """
#     Perform time series forecasting using PyCaret and plot the results.
#
#     Parameters:
#     data (pd.DataFrame): The input dataset with columns 'new_date' and 'count'.
#
#     Returns:
#     None
#     """
#     data['Month'] = [i.month for i in data['new_date']]
#     data['Year'] = [i.year for i in data['new_date']]
#     data['Series'] = np.arange(1, len(data) + 1)
#     data_pycaret = data[['Series', 'Year', 'Month', 'count']]
#
#
#     train = data_pycaret[data_pycaret['Year'] < 2024]
#     test = data_pycaret[data_pycaret['Year'] >= 2024]
#
#     s = setup(data_split_shuffle=False,
#               data=train,
#               test_data=test,
#               target='count',
#               fold_strategy='timeseries',
#               numeric_features=['Year', 'Series'],
#               fold=3,
#               transform_target=True,
#               session_id=123
#               )
#     best_models = compare_models()
#     best = compare_models(sort='MAE')
#     print('Best Model Parameters:', best.get_params())
#
#     tuned_model = tune_model(best, optimize='MAE', verbose=True)
#     print('Tuned Model Parameters:', tuned_model.get_params())
#
#     compare_models([best, tuned_model], sort='MAE')
#
#     prediction_holdout = predict_model(tuned_model)
#     predictions = predict_model(tuned_model, data=data_pycaret)
#
#     data_pycaret['Predicted'] = predictions['prediction_label']
#
#     plt.plot(data_pycaret['Series'], data_pycaret['count'], label='Original', marker='o')
#     plt.plot(data_pycaret['Series'], data_pycaret['Predicted'], label='Predicted', linestyle='--', marker='x')
#     plt.title('Original vs. Predicted Values')
#     plt.xlabel('Time Series')
#     plt.ylabel('Count')
#     plt.legend()
#     plt.show()
#
#
# # revenue
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# from transform import TimeSeriesDifferencer, SquareRootTransformer, SimpleMovingAverage, EWMAOptimizer, \
#     BoxCoxTransformer, CustomStandardScaler, ARIMATransformer, RawData
# from model import ARIMASARIMAXModel, ProphetModel, ARIMASARIMAXModel_, ProphetModel_
# from statsmodels.tsa.stattools import adfuller, kpss
# from prophet import Prophet
# import statsmodels.api as sm
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.filters.hp_filter import hpfilter
#
# data = pd.read_csv('oppr_report_df_989702.csv', encoding='ISO-8859-1')
# filtered_data = data[data['stage_name_cat'] == 'Closed Won']
# # print(filtered_data)
# data['created_date_local_oppr'] = pd.to_datetime(data['created_date_local_oppr'], format='%Y-%m-%d %H:%M:%S', errors='coerce').dt.date
# data['close_date_local'] = pd.to_datetime(data['close_date_local'], errors='coerce')
# data.loc[data['close_date_local'].dt.year > 2024, 'close_date_local'] = data['created_date_local_oppr']
# dataset = data[['amount', 'close_date_local']].copy()
# dataset['date'] = dataset['close_date_local'].dt.date
# dataset.drop('close_date_local', axis=1, inplace=True)
# dataset['date'] = pd.to_datetime(dataset['date'])
# dataset['day'] = 1
# dataset['month'] = dataset['date'].dt.month
# dataset['year'] = dataset['date'].dt.year
# dataset['new_date'] = pd.to_datetime(dataset[['year', 'month', 'day']])
# grouped_data = dataset.groupby('new_date', as_index=False)['amount'].sum()
# grouped_data = grouped_data.drop(grouped_data.index[-1])
# grouped_data = grouped_data.drop_duplicates(subset='new_date')
# grouped_data = grouped_data.set_index('new_date')
# full_range = pd.date_range(start=grouped_data.index.min(), end=grouped_data.index.max(), freq='MS')
# grouped_data = grouped_data.reindex(full_range, fill_value=0)
# grouped_data = grouped_data.reset_index()
# grouped_data.rename(columns={'index': 'new_date'}, inplace=True)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# grouped_data_1=grouped_data[76:]
# grouped_data=grouped_data[42:77]
# print(grouped_data)
# # import pdb;pdb.set_trace()
# pd.reset_option('display.max_rows')
# pd.reset_option('display.max_columns')
# grouped_data['count_orignal']=grouped_data['amount']
# array_ar=np.array(grouped_data['amount'])
# decomposition = sm.tsa.seasonal_decompose(array_ar, model='additive',period=4) #additive or multiplicative is data specific
# fig = decomposition.plot()
# plt.show()
# grouped_data['seasons']=decomposition.seasonal
# trends = decomposition.trend
# trend_series = pd.Series(trends)
# trend_series.ffill(inplace=True)
# trend_series.bfill(inplace=True)
# trend_series.interpolate(method='linear', inplace=True)
# # trend_series.iloc[-4:] = 0  # Set the last 4 values to 0
# filled_trend = trend_series.to_numpy()
# grouped_data['trend_decomposed']=filled_trend
# residuals = decomposition.resid
# residuals_series = pd.Series(residuals)
# residuals_series.ffill(inplace=True)
# residuals_series.bfill(inplace=True)
# residuals_series.interpolate(method='linear', inplace=True)
# filled_residuals=residuals_series.to_numpy()
# grouped_data['residuals_decomposed']=filled_residuals
# print('residuals',residuals_series)
# print('trends',trend_series)
# print('seasons',grouped_data['seasons'])
# model = Prophet()
# grouped_data['amount']=np.sqrt(np.array(grouped_data['amount']))
# print(grouped_data['amount'])
# # grouped_data['amount'] = grouped_data['amount'].replace(-float('inf'), 0)
# import pdb;pdb.set_trace()
# #
# models = [(ARIMASARIMAXModel,{'seasonal':True, 'm':7, 'trace':True}),(ProphetModel_, {}),(ARIMASARIMAXModel_,{'seasonal':True, 'm':7, 'trace':True})]
# fit_args_map = {
#     ARIMASARIMAXModel: {'data': grouped_data, 'target_column': 'amount', 'future_periods': 5, 'exogenous_columns': None},
#     ProphetModel_: {'data': grouped_data, 'date_column': 'new_date', 'target_column': 'trend_decomposed'},
#     ARIMASARIMAXModel_: {'data': grouped_data, 'target_column': 'trend_decomposed', 'future_periods': 5, 'exogenous_columns': None}
# }
#
# for model_class, model_kwargs in models:
#     model = model_class(**model_kwargs)
#
#     fit_args = fit_args_map.get(model_class, {})
#
#     model.fit(**fit_args)
#     print(f" Model: {model_class.__name__}")
#     print()
#     if isinstance(model, ARIMASARIMAXModel):
#         predict_arima = model.forecast()
#     if isinstance(model, ProphetModel_):
#         predict_prophet_ = np.array(model.predict(periods=5, freq='MS'))
#     if isinstance(model, ARIMASARIMAXModel_):
#         predict_arima_trend = model.forecast()
#
# # print(predict_prophet)
# trends_arima=predict_arima_trend[0]
# predict_arima_trends = trends_arima.to_numpy()
# result_array = (np.array(predict_prophet_))
# series_data=predict_arima[0]
# arima_amount = ((series_data.to_numpy())**2)
# print('arima_trends',predict_arima_trends)
# print('prophet_trends',result_array)
# print('arima_amount',arima_amount)
# net_trends=np.add(predict_arima_trends,result_array)
# import pdb;pdb.set_trace()
# net_trends=(np.add(predict_arima_trends,result_array))/2
# combined_result=(np.add(net_trends,arima_amount))/1
# # combined_result=combined_result_arima
# print(combined_result)
#
#
# last_date = grouped_data['new_date'].max()
# new_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=5, freq='MS')
# all_dates = pd.concat([grouped_data['new_date'], pd.Series(new_dates)])
# all_dates = pd.to_datetime(all_dates).sort_values().reset_index(drop=True)
# extended_dates_array = all_dates.to_numpy()
#
# plt.plot(grouped_data['new_date'], grouped_data['count_orignal'], label='Original Data', color='black')
# plt.plot(grouped_data_1['new_date'], grouped_data_1['amount'], label='Original Data(test)', color='purple')
# # plt.plot(extended_dates_array, result_array, color='green', label='Prophet Predictions')
# # plt.plot(extended_dates_array, np_predict_arima, color='orange', label='Arima Predictions')
# plt.plot(extended_dates_array, combined_result, color='red', label='combined Predictions')
# plt.xlabel('Date')
# plt.ylabel('Count')
# plt.title('Original Data and Prophet Model Predictions')
# plt.legend()
# plt.grid(True)
#
# plt.show()
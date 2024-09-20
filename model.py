import numpy as np
import pandas as pd
import pmdarima as pm
from prophet import Prophet
from pmdarima.model_selection import train_test_split
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class ARIMASARIMAXModel:
    def __init__(self, seasonal=True, m=7, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True,
                 **kwargs):
        """
        Initialize the ARIMASARIMAXModel with given parameters.

        Parameters:
        - seasonal: Whether to use seasonal ARIMA (default: True).
        - m: The number of periods in each season (default: 7).
        - trace: Whether to print status on the fitting (default: True).
        - error_action: Action to take if an error occurs (default: 'ignore').
        - suppress_warnings: Whether to suppress warnings (default: True).
        - stepwise: Whether to use stepwise algorithm for model selection (default: True).
        - kwargs: Additional keyword arguments for the ARIMA model.
        """
        self.seasonal = seasonal
        self.m = m
        self.trace = trace
        self.error_action = error_action
        self.suppress_warnings = suppress_warnings
        self.stepwise = stepwise
        self.model_kwargs = kwargs
        self.model = None
        self.forecasted_values = None
        self.conf_int = None
        self.dates = None

    def fit(self, data, target_column, exogenous_columns=None, future_periods=12):
        self.data = data
        self.target_column = target_column
        self.exogenous_columns = exogenous_columns

        # Create a future dates DataFrame with zero values
        last_date = pd.to_datetime(data['new_date'].max())
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_periods, freq='MS')
        future_data = pd.DataFrame({
            'new_date': future_dates,
            'seasons': [0] * future_periods,
            'residuals_decomposed': [0]*future_periods,

        })

        # Combine original data with future data
        extended_data = pd.concat([data, future_data], ignore_index=True)


        # Fit auto_arima model on the extended dataset
        self.model = pm.auto_arima(
            y=extended_data[target_column],
            exogenous=extended_data[exogenous_columns] if exogenous_columns else None,
            seasonal=self.seasonal,
            m=self.m,
            trace=self.trace,
            error_action=self.error_action,
            suppress_warnings=self.suppress_warnings,
            stepwise=self.stepwise
        )

        # Save the dates for plotting
        self.dates = extended_data['new_date']
        # Get in-sample predictions
        in_sample_preds = self.model.predict_in_sample()

        # Save in-sample predictions
        self.in_sample_preds = in_sample_preds
        # import pdb;pdb.set_trace()
        # Summary of the model
        print(self.model.summary())

    def forecast(self):
        if self.model is None:
            raise ValueError("The model is not fitted yet. Call the fit method first.")

        # Predict future periods
        forecast_periods = len(self.dates) - len(self.data)
        out_of_sample_forecast, conf_int = self.model.predict(
            n_periods=forecast_periods,
            exogenous=None,
            return_conf_int=True
        )
        # np.concatenate([self.in_sample_preds, out_of_sample_forecast])
        self.forecasted_values = self.in_sample_preds
        # self.conf_int = np.vstack([np.full((len(self.in_sample_preds), 2), np.nan), conf_int])

        return self.forecasted_values, self.conf_int

    def plot_forecast(self):
        if self.forecasted_values is None:
            raise ValueError("You need to run the forecast method first.")

        plt.figure(figsize=(6, 6))

        # Plot the original data
        plt.plot(self.data['new_date'], self.data[self.target_column], label='Original Data', color='black')
        # Plot the ARIMA model predictions with the corresponding same-sized interval of the dates
        forecast_dates = self.dates[:len(self.forecasted_values)]
        plt.plot(forecast_dates, self.forecasted_values, color='green', label='ARIMA Predictions')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.title('Original Data and ARIMA Model Predictions')
        plt.legend()
        plt.show()



class ARIMASARIMAXModel_:
    def __init__(self, seasonal=True, m=7, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True,
                 **kwargs):
        """
        Initialize the ARIMASARIMAXModel with given parameters.

        Parameters:
        - seasonal: Whether to use seasonal ARIMA (default: True).
        - m: The number of periods in each season (default: 7).
        - trace: Whether to print status on the fitting (default: True).
        - error_action: Action to take if an error occurs (default: 'ignore').
        - suppress_warnings: Whether to suppress warnings (default: True).
        - stepwise: Whether to use stepwise algorithm for model selection (default: True).
        - kwargs: Additional keyword arguments for the ARIMA model.
        """
        self.seasonal = seasonal
        self.m = m
        self.trace = trace
        self.error_action = error_action
        self.suppress_warnings = suppress_warnings
        self.stepwise = stepwise
        self.model_kwargs = kwargs
        self.model = None
        self.forecasted_values = None
        self.conf_int = None
        self.dates = None

    def fit(self, data, target_column, exogenous_columns=None, future_periods=12):
        self.data = data
        self.target_column = target_column
        self.exogenous_columns = exogenous_columns

        # Create a future dates DataFrame with zero values
        last_date = pd.to_datetime(data['new_date'].max())
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_periods, freq='MS')
        future_data = pd.DataFrame({
            'new_date': future_dates,
            'residuals_decomposed': [0] * future_periods
        })

        # Combine original data with future data
        extended_data = pd.concat([data, future_data], ignore_index=True)

        # Fit auto_arima model on the extended dataset
        self.model = pm.auto_arima(
            y=extended_data[target_column],
            exogenous=extended_data[exogenous_columns] if exogenous_columns else None,
            seasonal=self.seasonal,
            m=self.m,
            trace=self.trace,
            error_action=self.error_action,
            suppress_warnings=self.suppress_warnings,
            stepwise=self.stepwise
        )

        # Save the dates for plotting
        self.dates = extended_data['new_date']
        # Get in-sample predictions
        in_sample_preds = self.model.predict_in_sample()

        # Save in-sample predictions
        self.in_sample_preds = in_sample_preds
        # import pdb;pdb.set_trace()
        # Summary of the model
        print(self.model.summary())

    def forecast(self):
        if self.model is None:
            raise ValueError("The model is not fitted yet. Call the fit method first.")

        # Predict future periods
        forecast_periods = len(self.dates) - len(self.data)
        out_of_sample_forecast, conf_int = self.model.predict(
            n_periods=forecast_periods,
            exogenous=None,
            return_conf_int=True
        )
        # np.concatenate([self.in_sample_preds, out_of_sample_forecast])
        self.forecasted_values = self.in_sample_preds
        # self.conf_int = np.vstack([np.full((len(self.in_sample_preds), 2), np.nan), conf_int])

        return self.forecasted_values, self.conf_int

    def plot_forecast(self):
        if self.forecasted_values is None:
            raise ValueError("You need to run the forecast method first.")

        plt.figure(figsize=(6, 6))

        # Plot the original data
        plt.plot(self.data['new_date'], self.data[self.target_column], label='Original Data', color='black')
        # Plot the ARIMA model predictions with the corresponding same-sized interval of the dates
        forecast_dates = self.dates[:len(self.forecasted_values)]
        plt.plot(forecast_dates, self.forecasted_values, color='green', label='ARIMA Predictions')

        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.title('Original Data and ARIMA Model Predictions')
        plt.legend()
        plt.show()

class ARIMASARIMAXModel__:
    def __init__(self, seasonal=True, m=7, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True,
                 **kwargs):
        """
        Initialize the ARIMASARIMAXModel with given parameters.

        Parameters:
        - seasonal: Whether to use seasonal ARIMA (default: True).
        - m: The number of periods in each season (default: 7).
        - trace: Whether to print status on the fitting (default: True).
        - error_action: Action to take if an error occurs (default: 'ignore').
        - suppress_warnings: Whether to suppress warnings (default: True).
        - stepwise: Whether to use stepwise algorithm for model selection (default: True).
        - kwargs: Additional keyword arguments for the ARIMA model.
        """
        self.seasonal = seasonal
        self.m = m
        self.trace = trace
        self.error_action = error_action
        self.suppress_warnings = suppress_warnings
        self.stepwise = stepwise
        self.model_kwargs = kwargs
        self.model = None
        self.forecasted_values = None
        self.conf_int = None
        self.dates = None

    def fit(self, data, target_column, exogenous_columns=None, future_periods=12):
        self.data = data
        self.target_column = target_column
        self.exogenous_columns = exogenous_columns

        # Create a future dates DataFrame with zero values
        last_date = pd.to_datetime(data['new_date'].max())
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_periods, freq='MS')
        future_data = pd.DataFrame({
            'new_date': future_dates,
            'amount': [0] * future_periods,

        })

        # Combine original data with future data
        extended_data = pd.concat([data, future_data], ignore_index=True)


        # Fit auto_arima model on the extended dataset
        self.model = pm.auto_arima(
            y=extended_data[target_column],
            exogenous=extended_data[exogenous_columns] if exogenous_columns else None,
            seasonal=self.seasonal,
            m=self.m,
            trace=self.trace,
            error_action=self.error_action,
            suppress_warnings=self.suppress_warnings,
            stepwise=self.stepwise
        )

        # Save the dates for plotting
        self.dates = extended_data['new_date']
        # Get in-sample predictions
        in_sample_preds = self.model.predict_in_sample()

        # Save in-sample predictions
        self.in_sample_preds = in_sample_preds
        # import pdb;pdb.set_trace()
        # Summary of the model
        print(self.model.summary())

    def forecast(self):
        if self.model is None:
            raise ValueError("The model is not fitted yet. Call the fit method first.")

        # Predict future periods
        forecast_periods = len(self.dates) - len(self.data)
        out_of_sample_forecast, conf_int = self.model.predict(
            n_periods=forecast_periods,
            exogenous=None,
            return_conf_int=True
        )
        # np.concatenate([self.in_sample_preds, out_of_sample_forecast])
        self.forecasted_values = self.in_sample_preds
        # self.conf_int = np.vstack([np.full((len(self.in_sample_preds), 2), np.nan), conf_int])

        return self.forecasted_values, self.conf_int

    def plot_forecast(self):
        if self.forecasted_values is None:
            raise ValueError("You need to run the forecast method first.")

        plt.figure(figsize=(6, 6))

        # Plot the original data
        plt.plot(self.data['new_date'], self.data[self.target_column], label='Original Data', color='black')
        # Plot the ARIMA model predictions with the corresponding same-sized interval of the dates
        forecast_dates = self.dates[:len(self.forecasted_values)]
        plt.plot(forecast_dates, self.forecasted_values, color='green', label='ARIMA Predictions')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.title('Original Data and ARIMA Model Predictions')
        plt.legend()
        plt.show()

class ARIMASARIMAXModel___:
    def __init__(self, seasonal=True, m=7, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True,
                 **kwargs):
        """
        Initialize the ARIMASARIMAXModel with given parameters.

        Parameters:
        - seasonal: Whether to use seasonal ARIMA (default: True).
        - m: The number of periods in each season (default: 7).
        - trace: Whether to print status on the fitting (default: True).
        - error_action: Action to take if an error occurs (default: 'ignore').
        - suppress_warnings: Whether to suppress warnings (default: True).
        - stepwise: Whether to use stepwise algorithm for model selection (default: True).
        - kwargs: Additional keyword arguments for the ARIMA model.
        """
        self.seasonal = seasonal
        self.m = m
        self.trace = trace
        self.error_action = error_action
        self.suppress_warnings = suppress_warnings
        self.stepwise = stepwise
        self.model_kwargs = kwargs
        self.model = None
        self.forecasted_values = None
        self.conf_int = None
        self.dates = None

    def fit(self, data, target_column, exogenous_columns=None, future_periods=12):
        self.data = data
        self.target_column = target_column
        self.exogenous_columns = exogenous_columns

        # Create a future dates DataFrame with zero values
        last_date = pd.to_datetime(data['new_date'].max())
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_periods, freq='MS')
        future_data = pd.DataFrame({
            'new_date': future_dates,
            'trend_decomposed': [0] * future_periods,

        })

        # Combine original data with future data
        extended_data = pd.concat([data, future_data], ignore_index=True)


        # Fit auto_arima model on the extended dataset
        self.model = pm.auto_arima(
            y=extended_data[target_column],
            exogenous=extended_data[exogenous_columns] if exogenous_columns else None,
            seasonal=self.seasonal,
            m=self.m,
            trace=self.trace,
            error_action=self.error_action,
            suppress_warnings=self.suppress_warnings,
            stepwise=self.stepwise
        )

        # Save the dates for plotting
        self.dates = extended_data['new_date']
        # Get in-sample predictions
        in_sample_preds = self.model.predict_in_sample()

        # Save in-sample predictions
        self.in_sample_preds = in_sample_preds
        # import pdb;pdb.set_trace()
        # Summary of the model
        print(self.model.summary())

    def forecast(self):
        if self.model is None:
            raise ValueError("The model is not fitted yet. Call the fit method first.")

        # Predict future periods
        forecast_periods = len(self.dates) - len(self.data)
        out_of_sample_forecast, conf_int = self.model.predict(
            n_periods=forecast_periods,
            exogenous=None,
            return_conf_int=True
        )
        # np.concatenate([self.in_sample_preds, out_of_sample_forecast])
        self.forecasted_values = self.in_sample_preds
        # self.conf_int = np.vstack([np.full((len(self.in_sample_preds), 2), np.nan), conf_int])

        return self.forecasted_values, self.conf_int

    def plot_forecast(self):
        if self.forecasted_values is None:
            raise ValueError("You need to run the forecast method first.")

        plt.figure(figsize=(6, 6))

        # Plot the original data
        plt.plot(self.data['new_date'], self.data[self.target_column], label='Original Data', color='black')
        # Plot the ARIMA model predictions with the corresponding same-sized interval of the dates
        forecast_dates = self.dates[:len(self.forecasted_values)]
        plt.plot(forecast_dates, self.forecasted_values, color='green', label='ARIMA Predictions')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.title('Original Data and ARIMA Model Predictions')
        plt.legend()
        plt.show()

class ProphetModel:
    def __init__(self, **kwargs):
        """
        Initialize the Prophet model with given keyword arguments.

        Parameters:
        **kwargs: Arbitrary keyword arguments for the Prophet model.
        """
        self.model = Prophet(**kwargs)
        self.forecast = None

    def fit(self, data, date_column, target_column):
        # Prepare data for Prophet
        df_prophet = data[[date_column, target_column]].rename(columns={date_column: 'ds', target_column: 'y'})
        df_prophet['y'] = np.log(df_prophet['y'])  # Log transformation

        if np.isinf(df_prophet['y']).any():
            max_value = df_prophet['y'].replace([np.inf, -np.inf], np.nan).max()
            df_prophet['y'].replace([np.inf, -np.inf], max_value, inplace=True)
        self.model.fit(df_prophet)

    def predict(self, periods, freq='MS'):
        # Create future dataframe and make predictions
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        self.forecast = self.model.predict(future)

        # Inverse the log transformation
        self.forecast['trend'] = np.exp(self.forecast['trend'])
        self.forecast['yhat'] = np.exp(self.forecast['yhat'])
        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)
        # print(self.forecast)
        # pd.reset_option('display.max_rows')
        # pd.reset_option('display.max_columns')
        return self.forecast['yhat']
    def plot(self, original_data, date_column, target_column):
        # Merge original 'y' values into the forecast DataFrame
        original_data = original_data[[date_column, target_column]].rename(
            columns={date_column: 'ds', target_column: 'y'}
        )
        self.forecast = self.forecast.merge(original_data, on='ds', how='left')

        # # Inverse the log transformation of the original 'y' values
        # self.forecast['y'] = np.exp(self.forecast['y'])

        # Fill NaN values in the original 'y' column with 0
        self.forecast.fillna({'y':0},inplace=True)

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

        # Plot Original and Forecasted Counts
        ax1.plot(self.forecast['ds'], self.forecast['y'], label='Original y', marker='o')
        ax1.plot(self.forecast['ds'], self.forecast['yhat'], label='Forecast yhat', marker='x')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Count')
        ax1.set_title('Original and Forecasted Counts')
        ax1.set_ylim(0, self.forecast[['y', 'yhat']].max().max() + 10000)
        ax1.set_yticks(np.arange(0, self.forecast[['y', 'yhat']].max().max() + 10000, 10000))
        ax1.legend()

        # Plot Trend Component
        ax2.plot(self.forecast['ds'], self.forecast['trend'], label='Trend', color='green')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Trend')
        ax2.set_title('Trend Component')
        ax2.legend()

        plt.tight_layout()
        plt.show()

class ProphetModel_:
    def __init__(self, **kwargs):
        """
        Initialize the Prophet model with given keyword arguments.

        Parameters:
        **kwargs: Arbitrary keyword arguments for the Prophet model.
        """
        self.model = Prophet(**kwargs)
        self.forecast = None

    def fit(self, data, date_column, target_column):
        # Prepare data for Prophet
        df_prophet = data[[date_column, target_column]].rename(columns={date_column: 'ds', target_column: 'y'})
        df_prophet['y'] = np.sqrt(df_prophet['y'])  # sqrt transformation

        if np.isinf(df_prophet['y']).any():
            max_value = df_prophet['y'].replace([np.inf, -np.inf], np.nan).max()
            df_prophet['y'].replace([np.inf, -np.inf], max_value, inplace=True)
        self.model.fit(df_prophet)

    def predict(self, periods, freq='MS'):
        # Create future dataframe and make predictions
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        self.forecast = self.model.predict(future)

        # Inverse the log transformation
        self.forecast['trend'] = (self.forecast['trend'])**2
        self.forecast['yhat'] = (self.forecast['yhat'])**2

        return self.forecast['yhat']
    def plot(self, original_data, date_column, target_column):
        # Merge original 'y' values into the forecast DataFrame
        original_data = original_data[[date_column, target_column]].rename(
            columns={date_column: 'ds', target_column: 'y'}
        )
        self.forecast = self.forecast.merge(original_data, on='ds', how='left')
        self.forecast.fillna({'y':0},inplace=True)

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

        # Plot Original and Forecasted Counts
        ax1.plot(self.forecast['ds'], self.forecast['y'], label='Original y', marker='o')
        ax1.plot(self.forecast['ds'], self.forecast['yhat'], label='Forecast yhat', marker='x')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Count')
        ax1.set_title('Original and Forecasted Counts')
        ax1.set_ylim(0, self.forecast[['y', 'yhat']].max().max() + 10000)
        ax1.set_yticks(np.arange(0, self.forecast[['y', 'yhat']].max().max() + 10000, 10000))
        ax1.legend()

        # Plot Trend Component
        ax2.plot(self.forecast['ds'], self.forecast['trend'], label='Trend', color='green')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Trend')
        ax2.set_title('Trend Component')
        ax2.legend()

        plt.tight_layout()
        plt.show()


class ProphetModel__:
    def __init__(self, **kwargs):
        """
        Initialize the Prophet model with given keyword arguments.

        Parameters:
        **kwargs: Arbitrary keyword arguments for the Prophet model.
        """
        self.model = Prophet(**kwargs)
        self.forecast = None

    def fit(self, data, date_column, target_column):
        # Prepare data for Prophet
        df_prophet = data[[date_column, target_column]].rename(columns={date_column: 'ds', target_column: 'y'})
        df_prophet['y'] = (df_prophet['y'])

        if np.isinf(df_prophet['y']).any():
            max_value = df_prophet['y'].replace([np.inf, -np.inf], np.nan).max()
            df_prophet['y'].replace([np.inf, -np.inf], max_value, inplace=True)
        self.model.fit(df_prophet)

    def predict(self, periods, freq='MS'):
        # Create future dataframe and make predictions
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        self.forecast = self.model.predict(future)

        # Inverse the log transformation
        self.forecast['trend'] = (self.forecast['trend'])
        self.forecast['yhat'] = (self.forecast['yhat'])
        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)
        # print(self.forecast)
        # pd.reset_option('display.max_rows')
        # pd.reset_option('display.max_columns')
        return self.forecast['yhat']
    def plot(self, original_data, date_column, target_column):
        # Merge original 'y' values into the forecast DataFrame
        original_data = original_data[[date_column, target_column]].rename(
            columns={date_column: 'ds', target_column: 'y'}
        )
        self.forecast = self.forecast.merge(original_data, on='ds', how='left')

        # # Inverse the log transformation of the original 'y' values
        # self.forecast['y'] = np.exp(self.forecast['y'])

        # Fill NaN values in the original 'y' column with 0
        self.forecast.fillna({'y':0},inplace=True)

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

        # Plot Original and Forecasted Counts
        ax1.plot(self.forecast['ds'], self.forecast['y'], label='Original y', marker='o')
        ax1.plot(self.forecast['ds'], self.forecast['yhat'], label='Forecast yhat', marker='x')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Count')
        ax1.set_title('Original and Forecasted Counts')
        ax1.set_ylim(0, self.forecast[['y', 'yhat']].max().max() + 10000)
        ax1.set_yticks(np.arange(0, self.forecast[['y', 'yhat']].max().max() + 10000, 10000))
        ax1.legend()

        # Plot Trend Component
        ax2.plot(self.forecast['ds'], self.forecast['trend'], label='Trend', color='green')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Trend')
        ax2.set_title('Trend Component')
        ax2.legend()

        plt.tight_layout()
        plt.show()
# class ProphetForecasting(BaseEstimator, TransformerMixin):
#     def __init__(self, periods=0, add_seasonality=True, add_quarterly_seasonality=True):
#         self.periods = periods
#         self.add_seasonality = add_seasonality
#         self.add_quarterly_seasonality = add_quarterly_seasonality
#         self.model = Prophet()
#         if self.add_seasonality:
#             self.model.add_seasonality(name='daily', period=1, fourier_order=3)
#         if self.add_quarterly_seasonality:
#             self.model.add_seasonality(name='quarterly', period=90.25, fourier_order=10)
#         self.forecast = None
#
#     def fit(self, X, y=None):
#         # Check the columns before fitting
#         if 'ds' not in X.columns or 'y' not in X.columns:
#             raise ValueError('Dataframe must have columns "ds" and "y" with the dates and values respectively.')
#         self.model.fit(X)
#         return self
#
#     def transform(self, X):
#         future_dates = self.model.make_future_dataframe(periods=self.periods)
#         self.forecast = self.model.predict(future_dates)
#
#         seasonality_columns = ['yearly', 'weekly', 'daily','quarterly']
#         existing_columns = [col for col in seasonality_columns if col in self.forecast.columns]
#         if existing_columns:
#             self.forecast['seasonality'] = self.forecast[existing_columns].sum(axis=1)
#         else:
#             self.forecast['seasonality'] = 0
#
#         result_df = self.forecast[['ds', 'yhat', 'seasonality', 'trend']]
#         result_df.columns = ['ds', 'y', 'seasonality', 'trend']
#         return result_df



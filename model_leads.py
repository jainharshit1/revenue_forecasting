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
        last_date = pd.to_datetime(data['new_date'].max())
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_periods, freq='MS')
        future_data = pd.DataFrame({
            'new_date': future_dates,
            'seasons': [0] * future_periods,
            'residuals_decomposed': [0]*future_periods,

        })
        extended_data = pd.concat([data, future_data], ignore_index=True)
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
        self.dates = extended_data['new_date']
        in_sample_preds = self.model.predict_in_sample()
        self.in_sample_preds = in_sample_preds
        print(self.model.summary())

    def forecast(self):
        if self.model is None:
            raise ValueError("The model is not fitted yet. Call the fit method first.")

        forecast_periods = len(self.dates) - len(self.data)
        out_of_sample_forecast, conf_int = self.model.predict(
            n_periods=forecast_periods,
            exogenous=None,
            return_conf_int=True
        )
        self.forecasted_values = self.in_sample_preds
        return self.forecasted_values, self.conf_int

    def plot_forecast(self):
        if self.forecasted_values is None:
            raise ValueError("You need to run the forecast method first.")
        plt.figure(figsize=(6, 6))
        plt.plot(self.data['new_date'], self.data[self.target_column], label='Original Data', color='black')
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
        last_date = pd.to_datetime(data['new_date'].max())
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_periods, freq='MS')
        future_data = pd.DataFrame({
            'new_date': future_dates,
            'residuals_decomposed': [0] * future_periods
        })
        extended_data = pd.concat([data, future_data], ignore_index=True)
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
        self.dates = extended_data['new_date']
        in_sample_preds = self.model.predict_in_sample()
        self.in_sample_preds = in_sample_preds
        print(self.model.summary())

    def forecast(self):
        if self.model is None:
            raise ValueError("The model is not fitted yet. Call the fit method first.")

        forecast_periods = len(self.dates) - len(self.data)
        out_of_sample_forecast, conf_int = self.model.predict(
            n_periods=forecast_periods,
            exogenous=None,
            return_conf_int=True
        )
        self.forecasted_values = self.in_sample_preds
        return self.forecasted_values, self.conf_int

    def plot_forecast(self):
        if self.forecasted_values is None:
            raise ValueError("You need to run the forecast method first.")
        plt.figure(figsize=(6, 6))
        plt.plot(self.data['new_date'], self.data[self.target_column], label='Original Data', color='black')
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
        self.forecast['trend'] = np.exp(self.forecast['trend'])
        self.forecast['yhat'] = np.exp(self.forecast['yhat'])

        return self.forecast['yhat']
    def plot(self, original_data, date_column, target_column):
        original_data = original_data[[date_column, target_column]].rename(
            columns={date_column: 'ds', target_column: 'y'}
        )
        self.forecast = self.forecast.merge(original_data, on='ds', how='left')
        self.forecast.fillna({'y':0},inplace=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
        ax1.plot(self.forecast['ds'], self.forecast['y'], label='Original y', marker='o')
        ax1.plot(self.forecast['ds'], self.forecast['yhat'], label='Forecast yhat', marker='x')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Count')
        ax1.set_title('Original and Forecasted Counts')
        ax1.set_ylim(0, self.forecast[['y', 'yhat']].max().max() + 10000)
        ax1.set_yticks(np.arange(0, self.forecast[['y', 'yhat']].max().max() + 10000, 10000))
        ax1.legend()
        ax2.plot(self.forecast['ds'], self.forecast['trend'], label='Trend', color='green')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Trend')
        ax2.set_title('Trend Component')
        ax2.legend()
        plt.tight_layout()
        plt.show()

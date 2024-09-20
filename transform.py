import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA


class YourTransformClass:
    def __init__(self, **kwargs):
        """
        Initialize the transformer with specified parameters.
        """
        self.scaler = StandardScaler(**kwargs)

    def fit(self, data):
        """
        Fit the scaler to the numeric data.
        """
        numeric_data = data.select_dtypes(include=[float, int])  # Filter numeric columns
        self.scaler.fit(numeric_data)

    def transform(self, data):
        """
        Transform the numeric data using the fitted scaler.
        """
        numeric_data = data.select_dtypes(include=[float, int])  # Filter numeric columns
        return pd.DataFrame(self.scaler.transform(numeric_data), index=data.index, columns=numeric_data.columns)

    def fit_transform(self, data):
        """
        Fit the scaler and transform the numeric data.
        """
        numeric_data = data.select_dtypes(include=[float, int])  # Filter numeric columns
        return pd.DataFrame(self.scaler.fit_transform(numeric_data), index=data.index, columns=numeric_data.columns)


class TimeSeriesDifferencer:
    def __init__(self, **kwargs):
        self.k_diff = kwargs.get('k_diff', 1)  # Default to 1 if not provided
        self.fill_na_value = kwargs.get('fill_na_value', None)  # Default to None if not provided
        self.df = None
        self.column = None

    def fit(self, data, column):
        """
        Fit the differencer to the data.

        Parameters:
        data (array-like or DataFrame): The input data to fit the differencer on.
        column (str): The name of the column to apply differencing to.
        """
        self.df = pd.DataFrame(data)
        self.column = column
        self._apply_diff()

    def transform(self, data=None):
        """
        Transform the data using the fitted differencer.

        Parameters:
        data (array-like or DataFrame, optional): The data to transform. If not provided, will use the fitted data.

        Returns:
        DataFrame: The differenced data.
        """
        if data is not None:
            self.df = pd.DataFrame(data)
            self._apply_diff()
        return self.df

    def fit_transform(self, data, column):
        """
        Fit the differencer and transform the data in one step.

        Parameters:
        data (array-like or DataFrame): The input data to fit and transform.
        column (str): The name of the column to apply differencing to.

        Returns:
        DataFrame: The differenced data.
        """
        self.fit(data, column)
        return self.transform()

    def _apply_diff(self):
        self.df['differenced'] = self.df[self.column].diff(periods=self.k_diff)

    def get_differenced_data(self):
        """
        Get the differenced data.

        Returns:
        DataFrame: The differenced data.
        """
        return self.df

    def drop_na(self):
        """
        Drop NA values from the differenced data.

        Returns:
        DataFrame: The differenced data with NA values dropped.
        """
        return self.df.dropna()

    def fill_na(self, value=None):
        """
        Fill NA values in the differenced data.

        Parameters:
        value (optional): The value to fill NA values with. If not provided, will use the default fill_na_value.

        Returns:
        DataFrame: The differenced data with NA values filled.
        """
        if value is None:
            value = self.fill_na_value
        return self.df.fillna(value)


class SquareRootTransformer:
    def __init__(self, **kwargs):
        self.shift_negative = kwargs.get('shift_negative', True)
        self.shift_value = kwargs.get('shift_value', None)
        self.fitted = False

    def fit(self, data, column=None):
        """
        Fit the transformer to the specified column of the data.

        Parameters:
        - data: The data to fit the transformer to (array-like or pandas DataFrame/Series).
        - column: The column to apply the transformation to (if data is a DataFrame).
        """
        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("When data is a DataFrame, the 'column' parameter must be specified.")
            self.column = column
            data_column = data[column].values
        elif isinstance(data, pd.Series):
            self.column = None
            data_column = data.values
        else:
            self.column = None
            data_column = data

        self.min_value = np.min(data_column)

        if self.shift_negative and self.min_value < 0:
            if self.shift_value is None:
                self.shift_value = -self.min_value + 1
            else:
                self.shift_value = self.shift_value
        else:
            self.shift_value = 0

        self.fitted = True

    def transform(self, data):
        """
        Transform the data using the fitted transformer.

        Parameters:
        - data: The data to transform (array-like or pandas DataFrame/Series).

        Returns:
        - Transformed data (same type as input).
        """
        if not self.fitted:
            raise ValueError(
                "The transformer has not been fitted yet. Call 'fit' with appropriate data before using this method.")

        if isinstance(data, pd.DataFrame):
            if self.column is None:
                raise ValueError("When data is a DataFrame, the 'column' parameter must be specified.")
            data_column = data[self.column].values
            transformed_data_column = np.sqrt(data_column + self.shift_value)
            transformed_data = data.copy()
            transformed_data[self.column] = transformed_data_column
        elif isinstance(data, pd.Series):
            data_column = data.values
            transformed_data = np.sqrt(data_column + self.shift_value)
        else:
            data_column = data
            transformed_data = np.sqrt(data_column + self.shift_value)

        return transformed_data

    def fit_transform(self, data, column=None):
        """
        Fit the transformer and transform the data in one step.

        Parameters:
        - data: The data to fit and transform (array-like or pandas DataFrame/Series).
        - column: The column to apply the transformation to (if data is a DataFrame).

        Returns:
        - Transformed data (same type as input).
        """
        self.fit(data, column)
        return self.transform(data)


class SimpleMovingAverage:
    def __init__(self, **kwargs):
        """
        Initialize the SimpleMovingAverage class with specified parameters.

        Parameters:
        - kwargs: Arbitrary keyword arguments.
            - window_size (int): The number of periods to include in the moving average calculation.
        """
        self.window_size = kwargs.get('window_size', 5)  # Default window size to 5 if not provided

    def fit_transform(self, data, column_name):
        """
        Apply the simple moving average transformation to the specified column in the data.

        Parameters:
        data (pd.DataFrame): The input data containing the time series.
        column_name (str): The name of the column to apply the moving average on.

        Returns:
        pd.DataFrame: The original data with an additional column for the moving average.
        """
        data_copy = data.copy()
        data_copy[f'SMA_{self.window_size}'] = data_copy[column_name].rolling(window=self.window_size).mean()
        return data_copy


class EWMAOptimizer:
    def __init__(self, **kwargs):
        """
        Initialize the EWMA optimizer with specified parameters.

        Parameters:
        - kwargs: Arbitrary keyword arguments.
            - alpha_values: An array of alpha values to test for optimization (default: np.linspace(0.01, 1, 100)).
        """
        self.alpha_values = kwargs.get('alpha_values', np.linspace(0.01, 1, 100))
        self.best_alpha = None
        self.best_mse = float('inf')

    def time_based_split(self, data, test_size=0.2):
        """
        Split the data into train and test sets for time series analysis.

        Parameters:
        - data: DataFrame with time series data.
        - test_size: Proportion of the dataset to include in the test split.

        Returns:
        - train_data: Training data DataFrame.
        - test_data: Test data DataFrame.
        """
        # Ensure the data is sorted by date
        data = data.sort_values(by='new_date')

        # Calculate the split index
        split_index = int(len(data) * (1 - test_size))

        # Split the data
        train_data = data.iloc[:split_index]
        test_data = data.iloc[split_index:]

        return train_data, test_data

    def fit(self, train, test):
        """
        Fit the optimizer to the training and test data to find the best alpha.

        Parameters:
        - train: Training data (pandas Series or DataFrame).
        - test: Test data (pandas Series or DataFrame).
        """
        for alpha in self.alpha_values:
            mse = self._calculate_ewma_error(alpha, train, test)
            if mse < self.best_mse:
                self.best_mse = mse
                self.best_alpha = alpha

        print(f"Best alpha: {self.best_alpha}, with MSE: {self.best_mse}")

    def _calculate_ewma_error(self, alpha, train, test):
        numeric_train = train.select_dtypes(include=['number'])  # Select only numeric columns
        ewma_train = numeric_train.ewm(alpha=alpha, adjust=False).mean()

        # Assuming test also contains the same columns
        numeric_test = test.select_dtypes(include=['number'])  # Select only numeric columns
        ewma_test = numeric_test.ewm(alpha=alpha, adjust=False).mean()

        # Calculate the error (e.g., Mean Squared Error) between ewma_test and numeric_test
        mse = ((ewma_test - numeric_test) ** 2).mean().mean()  # Example error calculation
        return mse

    def transform(self, data):
        """
        Transform the data using the best alpha.

        Parameters:
        - data: Data to transform (pandas Series or DataFrame).

        Returns:
        - transformed_data: DataFrame with the EWMA applied.
        """
        if self.best_alpha is None:
            raise ValueError(
                "The optimizer has not been fitted yet. Call 'fit' with appropriate data before using this method.")

        numeric_data = data.select_dtypes(include=['number'])  # Select only numeric columns
        transformed_numeric_data = numeric_data.ewm(alpha=self.best_alpha, adjust=False).mean()

        # Combine transformed numeric data with the original non-numeric data
        non_numeric_data = data.select_dtypes(exclude=['number'])
        transformed_data = pd.concat([transformed_numeric_data, non_numeric_data], axis=1)

        return transformed_data

    def fit_transform(self, data, test_size=0.2):
        """
        Fit the optimizer and transform the data in one step.

        Parameters:
        - data: Data to transform (pandas Series or DataFrame).
        - test_size: Proportion of the dataset to include in the test split.

        Returns:
        - transformed_data: DataFrame with the EWMA applied.
        """
        train, test = self.time_based_split(data, test_size)
        self.fit(train, test)
        return self.transform(data)


class BoxCoxTransformer:
    def __init__(self, **kwargs):
        """
        Initialize the Box-Cox transformer with specified parameters.

        Parameters:
        - kwargs: Arbitrary keyword arguments.
            - lmbda_range: A tuple specifying the range of lambda values to test (default: (-5, 5)).
            - num_steps: Number of steps in the lambda search grid (default: 1000).
        """
        self.lmbda_range = kwargs.get('lmbda_range', (-5, 5))
        self.num_steps = kwargs.get('num_steps', 1000)
        self.lmbda = None

    def fit(self, data):
        """
        Fits the Box-Cox transformation by finding the optimal lambda.

        Parameters:
        - data: DataFrame, input data to be transformed.
        """
        numeric_data = data.select_dtypes(include=['number'])
        best_lambda = None
        best_log_likelihood = -np.inf
        for lmbda in np.linspace(self.lmbda_range[0], self.lmbda_range[1], self.num_steps):
            transformed_data = self._boxcox_transform(numeric_data, lmbda)
            log_likelihood = self._log_likelihood(transformed_data)
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_lambda = lmbda

        self.lmbda = best_lambda

    def transform(self, data):
        """
        Applies the Box-Cox transformation to the data using the fitted lambda.

        Parameters:
        - data: DataFrame, input data to be transformed.

        Returns:
        - DataFrame with transformed numeric data.
        """
        if self.lmbda is None:
            raise ValueError("The transformer has not been fitted yet.")
        numeric_data = data.select_dtypes(include=['number'])
        transformed_numeric_data = self._boxcox_transform(numeric_data, self.lmbda)

        # Combine transformed numeric data with the original non-numeric data
        non_numeric_data = data.select_dtypes(exclude=['number'])
        transformed_data = pd.concat([transformed_numeric_data, non_numeric_data], axis=1)

        return transformed_data

    def fit_transform(self, data):
        """
        Fits the Box-Cox transformation and applies it to the data.

        Parameters:
        - data: DataFrame, input data to be transformed.

        Returns:
        - DataFrame with transformed numeric data.
        """
        self.fit(data)

        return self.transform(data)

    def inverse_transform(self, transformed_data):
        """
        Applies the inverse Box-Cox transformation to the data using the fitted lambda.

        Parameters:
        - transformed_data: DataFrame, transformed data to be inverse transformed.

        Returns:
        - DataFrame with original numeric data.
        """
        if self.lmbda is None:
            raise ValueError("The transformer has not been fitted yet.")
        numeric_data = transformed_data.select_dtypes(include=['number'])
        original_numeric_data = self._inv_boxcox_transform(numeric_data, self.lmbda)

        # Combine inverse-transformed numeric data with the original non-numeric data
        non_numeric_data = transformed_data.select_dtypes(exclude=['number'])
        original_data = pd.concat([original_numeric_data, non_numeric_data], axis=1)

        return original_data

    def _boxcox_transform(self, data, lmbda):
        """
        Applies the Box-Cox transformation to the numeric data.

        Parameters:
        - data: DataFrame, numeric input data to be transformed.
        - lmbda: the lambda parameter for the transformation.

        Returns:
        - DataFrame with transformed numeric data.
        """
        if lmbda == 0:
            return np.log(data)
        else:
            return (data ** lmbda - 1) / lmbda

    def _inv_boxcox_transform(self, transformed_data, lmbda):
        """
        Applies the inverse Box-Cox transformation to the numeric data.

        Parameters:
        - transformed_data: DataFrame, numeric transformed data to be inverse transformed.
        - lmbda: the lambda parameter used for the transformation.

        Returns:
        - DataFrame with original numeric data.
        """
        if lmbda == 0:
            return np.exp(transformed_data)
        else:
            return (transformed_data * lmbda + 1) ** (1 / lmbda)

    def _log_likelihood(self, transformed_data):
        """
        Computes the log-likelihood of the transformed numeric data.

        Parameters:
        - transformed_data: DataFrame, transformed numeric data.

        Returns:
        - Scalar log-likelihood.
        """
        return -0.5 * np.sum((transformed_data.values - np.mean(transformed_data.values)) ** 2)


class CustomStandardScaler:
    def __init__(self, **kwargs):
        print(f"Initializing YourTransformClass with arguments: {kwargs}")
        # Remove 'data' from kwargs if it exists
        if 'data' in kwargs:
            kwargs.pop('data')
            print("Removed 'data' from kwargs")
        self.scaler = StandardScaler(**kwargs)

    def fit(self, data):
        self.check_numeric_data(data)
        numeric_data = data.select_dtypes(include=[float, int])
        self.scaler.fit(numeric_data)

    def transform(self, data):
        self.check_numeric_data(data)
        numeric_data = data.select_dtypes(include=[float, int])
        transformed_numeric_data = self.scaler.transform(numeric_data)
        non_numeric_data = data.select_dtypes(exclude=[float, int])
        transformed_data = pd.concat(
            [pd.DataFrame(transformed_numeric_data, index=data.index, columns=numeric_data.columns), non_numeric_data],
            axis=1)
        return transformed_data

    def fit_transform(self, data):
        self.check_numeric_data(data)
        numeric_data = data.select_dtypes(include=[float, int])
        transformed_numeric_data = self.scaler.fit_transform(numeric_data)
        non_numeric_data = data.select_dtypes(exclude=[float, int])
        transformed_data = pd.concat(
            [pd.DataFrame(transformed_numeric_data, index=data.index, columns=numeric_data.columns), non_numeric_data],
            axis=1)
        return transformed_data

    def check_numeric_data(self, data):
        non_numeric_cols = data.select_dtypes(exclude=[float, int]).columns
        if not non_numeric_cols.empty:
            print(f"Warning: Data contains non-numeric columns: {', '.join(non_numeric_cols)}")


class ARIMATransformer:
    def __init__(self, p, d, q, **kwargs):
        """
        Initialize the ARIMATransformer with given order (p, d, q).

        Parameters:
        p (int): The number of lag observations included in the model (AR).
        d (int): The number of times that the raw observations are differenced (I).
        q (int): The size of the moving average window (MA).
        kwargs: Arbitrary keyword arguments for the ARIMA model.
        """
        self.p = p
        self.d = d
        self.q = q
        self.model_kwargs = kwargs
        self.model = None
        self.fitted_model = None

    def fit(self, data, column):
        """
        Fit the ARIMA model to the data.

        Parameters:
        data (pd.DataFrame): The time series data to fit the model on.
        column (str): The column to fit the model on.
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data.")
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"The column '{column}' must be numeric to fit the ARIMA model.")

        self.model = ARIMA(data[column], order=(self.p, self.d, self.q), **self.model_kwargs)
        self.fitted_model = self.model.fit()

    def transform(self, data, column):
        """
        Apply the ARIMA transformation to the data.

        Parameters:
        data (pd.DataFrame): The time series data to transform.
        column (str): The column to transform.

        Returns:
        pd.DataFrame: The transformed time series data.
        """
        if self.fitted_model is None:
            raise ValueError("The model must be fitted before transforming data.")
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data.")

        transformed_data = self.fitted_model.predict(start=0, end=len(data) - 1, dynamic=False)
        result = data.copy()
        result[column] = transformed_data
        return result

    def forecast(self, steps):
        """
        Generate forecasts from the fitted ARIMA model.

        Parameters:
        steps (int): The number of steps to forecast into the future.

        Returns:
        array-like: The forecasted values.
        """
        if self.fitted_model is None:
            raise ValueError("The model must be fitted before generating forecasts.")

        forecast = self.fitted_model.forecast(steps=steps)
        return forecast

    def summary(self):
        """
        Get the summary of the fitted ARIMA model.

        Returns:
        str: The summary of the model.
        """
        if self.fitted_model is None:
            raise ValueError("The model must be fitted before getting the summary.")

        return self.fitted_model.summary()


class RawData:
    def __init__(self, **kwargs):
        """
        Initialize the RawData transformer with optional parameters.

        Parameters:
        - kwargs: Arbitrary keyword arguments.
        """
        # Store kwargs if needed for future extensions or logging purposes
        self.kwargs = kwargs

    def fit(self, data):
        """
        Fit method for the RawData transformer.

        Parameters:
        - data: DataFrame or array-like, data to fit the transformer.
        """
        # No fitting necessary for identity transformation
        pass

    def transform(self, data):
        """
        Transform method for the RawData transformer.

        Parameters:
        - data: DataFrame or array-like, data to transform.

        Returns:
        - DataFrame: The input data as a DataFrame.
        """
        return pd.DataFrame(data, columns=data.columns)

    def fit_transform(self, data):
        """
        Fit and transform method for the RawData transformer.

        Parameters:
        - data: DataFrame or array-like, data to fit and transform.

        Returns:
        - DataFrame: The input data as a DataFrame.
        """
        # No fitting necessary, simply return the input data as is
        return pd.DataFrame(data, columns=data.columns)

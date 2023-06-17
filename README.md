# Cybersapient_Assignment
# Gold Price Analysis
This code performs an analysis of gold price data using various techniques such as time series decomposition, stationarity testing, differencing, linear regression modeling, and evaluation.

# Required Libraries
The following libraries are imported to run the code:

pandas for data manipulation and analysis
numpy for numerical computations
matplotlib.pyplot for plotting
statsmodels.tsa.seasonal.seasonal_decompose for time series decomposition
statsmodels.tsa.stattools.adfuller for the augmented Dickey-Fuller test
statsmodels.graphics.tsaplots.plot_acf and statsmodels.graphics.tsaplots.plot_pacf for autocorrelation and partial autocorrelation plots
sklearn.model_selection.train_test_split for splitting the data into train and test sets
sklearn.linear_model.LinearRegression for linear regression modeling
sklearn.metrics.r2_score for calculating the R-squared score

# Loading and Preparing the Data
The OHLC (Open, High, Low, Close) data is loaded from an Excel file using pd.read_excel.
The 'Date' column is converted to datetime format using pd.to_datetime.
The 'Date' column is set as the index of the DataFrame using df.set_index('Date', inplace=True).
The DataFrame is sorted by date in ascending order using df.sort_index(ascending=True, inplace=True).
Optionally, you can resample the data to a different frequency by uncommenting the line df = df.resample('M').mean(). This line resamples the data to monthly frequency by taking the mean of each month's data.

# Visualizing the Data
The closing prices are plotted over time using plt.plot(df['Close']) to provide an overview of the gold price trends.

# Time Series Decomposition
The seasonal_decompose function is applied to the closing prices (df['Close']) to decompose the series into its trend, seasonality, and residual components.
A figure with four subplots is created to visualize the original data, trend, seasonality, and residuals using plt.subplots(4, 1, figsize=(12, 10)).
Each subplot is plotted using axes[i].plot() and labeled accordingly.
The plots are displayed using plt.tight_layout() and plt.show().

# Stationarity Testing
A function named adf_test is defined to perform the augmented Dickey-Fuller test for stationarity.
The augmented Dickey-Fuller test is applied to the closing prices (df['Close']) using adf_test(df['Close']).
Differencing is performed on the closing prices to make the series stationary (df['Differenced_Close'] = df['Close'].diff()).
The augmented Dickey-Fuller test is applied to the differenced series (adf_test(df['Differenced_Close'])).

#Linear Regression Modeling
Daily price changes are calculated by differencing the closing prices (df['Price_Change'] = df['Close'].diff()).
The first row (NaN) is removed since it has no price change (df = df[1:]).
The data is split into train and test sets using an 80-20 split ratio (train_data, test_data = df[:train_size], df[train_size:]).
The feature and target variables are prepared (X_train, y_train, X_test, y_test).
A linear regression model is created and trained using LinearRegression().
Predictions are made on the training and test sets using model.predict().
The R-squared scores are calculated for the training and test sets using r2_score().
The R-squared scores are printed.

# Model Evaluation
The actual vs. predicted price changes are plotted using plt.scatter(y_test, y_pred_test).
The residuals are calculated by subtracting the predicted price changes from the actual price changes (residuals = y_test - y_pred_test).
A residual plot is created by plotting the actual price changes against the residuals using plt.scatter(y_test, residuals).
A horizontal line at y=0 is added to the residual plot using plt.axhline(y=0, color='r', linestyle='--').
Both plots are displayed using plt.show().

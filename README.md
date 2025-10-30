# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 28.10.2025

### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing

data = pd.read_csv('gold.csv')

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

close_data = data[['Close']]
print("Shape of the dataset:", close_data.shape)
print("First 10 rows of the dataset:")
print(close_data.head(10))

plt.plot(close_data['Close'], label='Original Gold Close Data')
plt.title('Original Gold Close Data')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.show()

rolling_mean_5 = close_data['Close'].rolling(window=5).mean()
rolling_mean_10 = close_data['Close'].rolling(window=10).mean()

rolling_mean_5.head(10)
rolling_mean_10.head(20)

plt.plot(close_data['Close'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)')
plt.plot(rolling_mean_10, label='Moving Average (window=10)')
plt.title('Moving Average of Gold Close Data')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.show()

data_monthly = data.resample('MS').sum()
data_monthly_close = data_monthly['Close']

scaled_data = pd.Series(data_monthly.values.reshape(-1, 1).flatten())

scaled_data = scaled_data + 1
x = int(len(scaled_data) * 0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

ax = train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add", "test_data"])
ax.set_title('Visual evaluation')

model = ExponentialSmoothing(data_monthly_close, trend='add', seasonal='mul', seasonal_periods=12).fit()
predictions = model.forecast(steps=int(len(data_monthly_close)/4))
ax = data_monthly_close.plot()
predictions.plot(ax=ax)
ax.legend(["data_monthly_close", "predictions"])
ax.set_xlabel('Gold Close Price')
ax.set_ylabel('Months')
ax.set_title('Prediction')
```

### OUTPUT:

Moving Average

Plot Transform Dataset

Exponential Smoothing



### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.

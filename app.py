import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest

# Define the Streamlit app
st.title("Anomaly Detection and Time Series Forecasting Web Application") #It sets the website title to 'Anomaly Detection and Time Series Forecasting Web Application'

st.set_option('deprecation.showPyplotGlobalUse', False) #It hides all the warnings in python output

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"]) # A widget to let users input his/her CSV files.
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, parse_dates=['Month']) # Creates a panda's DataFrame object and loads the contents of CSV file in it, and considers the 'month' column as dates. 

    # Data Preprocessing
    st.subheader("Data Overview") # Sets the subheader to 'Data Overview'
    
    data_display = data.head().copy() # Creates a DataFrame object and stores the first few rows of data.
    
    data_display.columns = ['1', '2'] # Changes the column name of 'data_display' to '1' and '2' for simplification.
    
    data_display.index = ['' for _ in range(len(data_display))]  # Set index names to empty strings. Index names/values are unique labels attached to each row of the DataFrame.
    
    st.write(data_display) # Write's the DataFrame object as table onto the website.

    # Function to check stationarity using ADF Test
    def check_stationarity(passengers):
        result = adfuller(passengers) # adfuller() checks whether data is stationary or not and returns an array containing values such as statistic, p-value etc.
        
        adf_statistic = result[0] # Returns the test statistic value.
        
        p_value = result[1] # Returns the p-value.

        st.subheader("ADF Test for Stationarity") # Prints 'ADF Test for Stationarity' as subheading on the website.
        
        st.write('ADF Statistic:', adf_statistic) # Prints the test statistic value with label 'ADF Statistic:' on the website.
        
        st.write('p-value:', p_value) # Prints the test statistic p-value with label 'p-value:' on the website.

    # Function to plot time series
    def plot_time_series_with_index(time_index, passengers, title):
        st.subheader(title)
        plt.figure(figsize=(10, 6))
        plt.plot(time_index, passengers) # Plot values of 'time_index' and 'passenger' in combination of the form of (x,y) coordinaties.
        plt.xlabel('')
        plt.ylabel('')
        plt.title(title)
        st.pyplot()

    # Function to perform seasonal decomposition
    def decompose_seasonality(passengers):
        decomposition = seasonal_decompose(passengers, period=12) # seasonal_decompose() returns an object having trend, seasonal, and residual as attributes. 
                                                                                    # 'passenger' represents Series object containing passenger column data.
                                                                                    # 'period' = 12 means that the repetition of patterns are expected to be yearly.
        
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        st.subheader("Seasonal Decomposition")
        st.write("Original Data:")
        plot_time_series_with_index(data['Month'], passengers, "Original Data") # Passing data['month'] Series object as index values.
                                                                                # Passing 'passengers' Series object.
                                                                                # Passing 'Original Data' as title.
        
        st.write("Trend Component:")
        plot_time_series_with_index(data['Month'], trend, "Trend Component") # Passing data['month'] Series object as index values.
                                                                             # Passing 'trend' Series object containing trend component values.
                                                                             # Passing 'Trend Component' as title.
        
        st.write("Seasonal Component:")
        plot_time_series_with_index(data['Month'], seasonal, "Seasonal Component") # Passing data['month'] Series object as index values.
                                                                                   # Passing 'seasonal' Series object containing seasonal component values.
                                                                                   # Passing 'Seasonal Component' as title.
        
        st.write("Residual Component:")
        plot_time_series_with_index(data['Month'], residual, "Residual Component")  # Passing data['month'] Series object as index values.
                                                                                    # Passing 'residual' Series object containing residual component values.
                                                                                    # Passing 'Residual Component' as title.

    # Function to remove anomalies using Isolation Forest
    def remove_anomalies(passengers): 
        outlier_detector = IsolationForest() # Creates Isolation Forest Model object to detect anomalies.  
        
        outlier_detector.fit(passengers.values.reshape(-1, 1)) # Converts the 1D passengers data into 2D because Isolation Forest uses 2D input for training.
        
        anomalies = outlier_detector.predict(passengers.values.reshape(-1, 1)) # Uses 2D data to predict anomalies after training.
        
        data['anomaly'] = anomalies # Anomalies stored in 'anomalies' are then added as a column in the 'data' DataFrame object.

        st.subheader("Anomaly Detection and Removal")
        st.subheader("Data with Anomalies:")
        plt.figure(figsize=(10, 6))
        plt.plot(data['Month'], passengers, label='Original')
        plt.scatter(data[data['anomaly'] == -1]['Month'], passengers[data['anomaly'] == -1], color='r', label='Anomaly') # Marks anomalies as red dots.
        plt.xlabel('')
        plt.ylabel('')
        plt.title('Data with Anomalies')
        plt.legend()
        st.pyplot()

        clean_data = data[data['anomaly'] != -1] # New DataFrame object 'clean_data' is created with values of 'data' DataFrame object excluding anomaly values.

        st.subheader("Data Without Anomalies")
        plt.figure(figsize=(10, 6))
        plt.plot(clean_data['Month'], clean_data['#Passengers'], label='Clean Data')
        plt.xlabel('')
        plt.ylabel('')
        plt.title('Data Without Anomalies')
        st.pyplot()

    # Function to fit and forecast using ARIMA on data without anomalies
    def fit_forecast_arima_without_anomalies(clean_data):
        passengers = clean_data['#Passengers'] # Stores anomalies excluded '#passenger' column values in 'passenger' variable.
        
        model = ARIMA(passengers, order=(1, 1, 1)) # order(p=1,d=1,q=1) represents:
                                                   # p = 1 represents each value depending on the previous value.
                                                   # d = 1 represents the data that has been differenced once.
                                                   # q = 1 represents one past forecast error.
        
        model_fit = model.fit() # Trains ARIMA model upon the data and stores in 'model_fit'.
        
        num_periods = 12 # Representing 12 periods for which the prediction has to be done.
        
        forecast_index = pd.date_range(start=clean_data['Month'].iloc[-1], periods=num_periods + 1, freq='M')[1:] # Generates a dates range starting from the next date of the last date in the dataset to the date formed by specifying number of periods.
        
        forecast = model_fit.predict(start=len(passengers), end=len(passengers) + num_periods - 1) # Specifies number of values to be forcasted.

        st.subheader("ARIMA Model Forecast (Without Anomalies)")
        plt.figure(figsize=(10, 6))
        plt.plot(clean_data['Month'], clean_data['#Passengers'], label='Clean Data')
        
        plt.plot(forecast_index, forecast, label='Forecast') # Plots forcasted values in the combinations of 'forecast_index' and 'forecast' values.
        
        plt.xlabel('Month')
        plt.ylabel('#Passengers')
        plt.title('Data with Forecast (Without Anomalies)')
        plt.legend()
        st.pyplot()

    check_stationarity(data['#Passengers'])
    decompose_seasonality(data['#Passengers'])
    remove_anomalies(data['#Passengers'])

    # Call the modified function with data without anomalies
    clean_data = data[data['anomaly'] != -1]
    fit_forecast_arima_without_anomalies(clean_data)

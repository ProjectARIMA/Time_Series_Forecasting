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
st.title("Anomaly Detection and Time Series Forecasting Web Application")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, parse_dates=['Month'])

    # Data Preprocessing
    st.subheader("Data Overview")
    data_display = data.head().copy()
    data_display.columns = ['1', '2']
    data_display.index = ['' for _ in range(len(data_display))]  # Set index names to empty strings
    st.write(data_display)

    # Function to check stationarity using ADF Test
    def check_stationarity(passengers):
        result = adfuller(passengers)
        adf_statistic = result[0]
        p_value = result[1]

        st.subheader("ADF Test for Stationarity")
        st.write('ADF Statistic:', adf_statistic)
        st.write('p-value:', p_value)

    # Function to plot time series
    def plot_time_series_with_index(time_index, passengers, title):
        st.subheader(title)
        plt.figure(figsize=(10, 6))
        plt.plot(time_index, passengers)
        plt.xlabel('')
        plt.ylabel('')
        plt.title(title)
        st.pyplot()

    # Function to perform seasonal decomposition
    def decompose_seasonality(passengers):
        decomposition = seasonal_decompose(passengers, model='additive', period=12)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        st.subheader("Seasonal Decomposition")
        st.write("Original Data:")
        plot_time_series_with_index(data['Month'], passengers, "Original Data")
        st.write("Trend Component:")
        plot_time_series_with_index(data['Month'], trend, "Trend Component")
        st.write("Seasonal Component:")
        plot_time_series_with_index(data['Month'], seasonal, "Seasonal Component")
        st.write("Residual Component:")
        plot_time_series_with_index(data['Month'], residual, "Residual Component")

    # Function to remove anomalies using Isolation Forest
    def remove_anomalies(passengers):
        outlier_detector = IsolationForest(contamination=0.01)
        outlier_detector.fit(passengers.values.reshape(-1, 1))
        anomalies = outlier_detector.predict(passengers.values.reshape(-1, 1))
        data['anomaly'] = anomalies

        st.subheader("Anomaly Detection and Removal")
        st.subheader("Data with Anomalies:")
        plt.figure(figsize=(10, 6))
        plt.plot(data['Month'], passengers, label='Original')
        plt.scatter(data[data['anomaly'] == -1]['Month'], passengers[data['anomaly'] == -1], color='r', label='Anomaly')
        plt.xlabel('')
        plt.ylabel('')
        plt.title('Data with Anomalies')
        plt.legend()
        st.pyplot()

        clean_data = data[data['anomaly'] != -1]

        st.subheader("Data Without Anomalies")
        plt.figure(figsize=(10, 6))
        plt.plot(clean_data['Month'], clean_data['#Passengers'], label='Clean Data')
        plt.xlabel('')
        plt.ylabel('')
        plt.title('Data Without Anomalies')
        st.pyplot()

    # Function to fit and forecast using ARIMA
    def fit_forecast_arima(passengers):
        model = ARIMA(passengers, order=(1, 1, 1))
        model_fit = model.fit()
        num_periods = 12
        forecast_index = pd.date_range(start=data['Month'].iloc[-1], periods=num_periods + 1, freq='M')[1:]
        forecast = model_fit.predict(start=len(passengers), end=len(passengers) + num_periods - 1)
        st.subheader("ARIMA Model Forecast")
        plt.figure(figsize=(10, 6))
        plt.plot(data['Month'], passengers, label='Original')
        plt.plot(forecast_index, forecast, label='Forecast')
        plt.xlabel('')
        plt.ylabel('')
        plt.title('Data with Forecast')
        plt.legend()
        st.pyplot()

    check_stationarity(data['#Passengers'])
    decompose_seasonality(data['#Passengers'])
    remove_anomalies(data['#Passengers'])
    fit_forecast_arima(data['#Passengers'])

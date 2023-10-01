import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from statsmodels.stats.diagnostic import acorr_ljungbox

# Define the Streamlit app
def main():
    st.title("Air Passengers Forecasting App")

    # Upload a dataset
    uploaded_file = st.file_uploader("Upload a CSV file in the specified format", type=["csv"])
    
    if uploaded_file is not None:
        # Load the uploaded dataset
        data = pd.read_csv(uploaded_file, parse_dates=["Month"], index_col="Month")

        # Display the first few rows of the dataset
        st.subheader("Data Preview")
        st.write(data.head())

        # Plot the passenger counts
        st.subheader("Passenger Count Plot")
        st.line_chart(data["#Passengers"])

        # Perform seasonal decomposition
        result = seasonal_decompose(data["#Passengers"], model="multiplicative")

        # Plot the decomposition components
        st.subheader("Seasonal Decomposition")
        st.pyplot(result.plot())

        # Fit an auto ARIMA model
        model = auto_arima(data["#Passengers"], seasonal=True, m=12, suppress_warnings=True)

        # Get the residuals
        residuals = model.resid()

        # Perform Ljung-Box test
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)

        # Display Ljung-Box test results
        st.subheader("Ljung-Box Test Results")
        st.write(lb_test)

        # Forecast future values
        forecast, conf_int = model.predict(n_periods=12, return_conf_int=True)

        # Create a dataframe to store the forecasted values and confidence intervals
        forecast_df = pd.DataFrame({'Forecast': forecast, 'Lower Bound': conf_int[:, 0], 'Upper Bound': conf_int[:, 1]}, index=pd.date_range(start=data.index[-1], periods=12, freq="M"))

        # Plot the original data and the forecasted values
        st.subheader("Air Passengers Forecast")
        st.line_chart(data["#Passengers"], use_container_width=True)
        st.line_chart(forecast_df, use_container_width=True)

if __name__ == "__main__":
    main()

""" 
Python Script to generate Spare Parts Demand for 14th Month by 13 Last Month
Original Author     : Brandon
Original Modified   : 2024-12-24
Modified By         : Unknown Player
Modified Date       : 2024-12-24
"""

# %%
# PACKAGES
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.metrics import mean_squared_error, r2_score  # For model evaluation metrics
from sklearn.linear_model import LinearRegression  # For linear regression modeling
from sklearn.preprocessing import PolynomialFeatures  # For polynomial regression modeling
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing  # For time series forecasting
from tabulate import tabulate  # For tabular display of data
import os  # For interacting with the operating system
import json  # For handling JSON data
import sys
import requests
import math

# %%
# Retrive JSON Data From API.
url = "http://172.16.1.59:18080/v1/web/get-spare-parts-all-history-data"

# Fetch JSON data from the API
response = requests.get(url)
response.raise_for_status()  # Raise an error if the request fails
df = response.json()  # Parse JSON data

# Convert JSON to Pandas DataFrame
data = pd.DataFrame(df['data'])
print(data)

# %%



# %%
# FUNCTION TO PROCESS DATA FOR A SPECIFIC P/N
def process_pn(data, pn_to_filter):
    # Filter data for the specified part number (P/N)
    filtered_data = data[data['P/N'] == pn_to_filter]
    formatted_data = []  # Initialize an empty list for formatted data

    # Loop through filtered rows to reformat data
    for index, row in filtered_data.iterrows():
        # Extract quantities from D-1 to D-13 and reverse the order
        quantities = [row[f'D-{i}'] for i in range(2, 14)][::-1]  # Reverse here

        # Create formatted data with period and quantity
        for period, quantity in enumerate(quantities, start=1):
            formatted_data.append({
                'P/N': str(row['P/N']),  # Ensure P/N is stored as a string
                'Period': period,  # Add period index
                'Quantity': quantity if not pd.isna(quantity) else None  # Handle missing data
            })

    # Convert the list of dictionaries to a DataFrame
    formatted_df = pd.DataFrame(formatted_data)

    # Calculate mean and standard deviation, and set upper bound
    mean_quantity = formatted_df['Quantity'].mean()
    std_quantity = formatted_df['Quantity'].std()
    LB = 0  # Lower bound is set to 0
    UB = mean_quantity + 1.5 * std_quantity  # Calculate upper bound as mean + 1.5 * std deviation
    mean_quantity = round(mean_quantity, 4)  # Round mean value
    std_quantity = round(std_quantity, 4)  # Round standard deviation
    UB = round(UB, 4)  # Round upper bound

    # Print metrics in a table format
    table = [
        ["Mean Quantity", mean_quantity],
        ["Standard Deviation", std_quantity],
        ["Upper Bound (UB)", UB]
    ]
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="pretty"))

    # Clip quantities to upper bound (replace values above UB with UB)
    formatted_df['Quantity'] = np.clip(formatted_df['Quantity'], LB, UB)
    return formatted_df  # Return the processed DataFrame

# SELECT P/N TO FILTER
pn_to_filter = 5372426  # Change this to filter data for a different part number
formatted_df = process_pn(data, pn_to_filter)  # Process data for the selected P/N

# MOVING AVERAGE
# Initialize Period column and moving average prediction column
formatted_df['Period'] = np.arange(1, len(formatted_df) + 1)
formatted_df['MA_Prediction'] = np.nan

# Calculate moving average for each period
for i in range(1, len(formatted_df) + 1):
    formatted_df.loc[formatted_df['Period'] == i, 'MA_Prediction'] = formatted_df['Quantity'].iloc[:i].mean()

# If there are 12 periods, predict the 13th period as the mean of the last 12 quantities
if len(formatted_df) == 12:
    last_12_mean = formatted_df['Quantity'].iloc[-12:].mean()
    formatted_df = pd.concat([
        formatted_df,
        pd.DataFrame({'Period': [13], 'MA_Prediction': [last_12_mean], 'Quantity': [np.nan]})
    ], ignore_index=True)

# WEIGHTED MOVING AVERAGE (WMA)
def calculate_exponential_weighted_moving_average(values, alpha_wma=0.3):
    # Calculate exponential weighted moving average with specified alpha
    return values.ewm(alpha=alpha_wma, adjust=False).mean()

alpha_wma = 0.3  # Smoothing parameter for WMA
formatted_df['Smoothed_Weighted_Prediction'] = calculate_exponential_weighted_moving_average(
    formatted_df['Quantity'], alpha_wma
)

# LINEAR REGRESSION
# Prepare data for linear regression
valid_data = formatted_df.dropna(subset=['Quantity'])
X = valid_data[['Period']]  # Independent variable
y = valid_data['Quantity']  # Dependent variable
model = LinearRegression()  # Initialize linear regression model
model.fit(X, y)  # Fit model to the data
formatted_df['Linear_Regression_Prediction'] = model.predict(formatted_df[['Period']])  # Predict for all periods

# POLYNOMIAL REGRESSION (DEGREE 2)
degree_2 = 2  # Degree of the polynomial
poly = PolynomialFeatures(degree=degree_2)  # Create polynomial features
X_poly = poly.fit_transform(X)  # Transform input features
poly_model = LinearRegression()  # Initialize linear regression model
poly_model.fit(X_poly, y)  # Fit polynomial model
X_all_poly = poly.transform(formatted_df[['Period']])  # Transform all periods
formatted_df['Polynomial_Regression_Prediction_Degree_2'] = poly_model.predict(X_all_poly)  # Predict values

# POLYNOMIAL REGRESSION (DEGREE 3)
degree_3 = 3  # Degree of the polynomial
poly = PolynomialFeatures(degree=degree_3)  # Create polynomial features
X_poly = poly.fit_transform(X)  # Transform input features
poly_model = LinearRegression()  # Initialize linear regression model
poly_model.fit(X_poly, y)  # Fit polynomial model
X_all_poly = poly.transform(formatted_df[['Period']])  # Transform all periods
formatted_df['Polynomial_Regression_Prediction_Degree_3'] = poly_model.predict(X_all_poly)  # Predict values

# SINGLE EXPONENTIAL SMOOTHING (SES)
alpha_ses = 0.8  # Smoothing parameter for SES
def apply_ses(values, alpha=alpha_ses):
    # Apply single exponential smoothing to the data
    ses_model = SimpleExpSmoothing(values).fit(smoothing_level=alpha, optimized=False)
    return ses_model.fittedvalues

# DOUBLE EXPONENTIAL SMOOTHING (DES)
beta_des = 0.3  # Trend smoothing parameter for DES
def apply_des(values, alpha=alpha_ses, beta=beta_des):
    # Apply double exponential smoothing to the data
    des_model = ExponentialSmoothing(values, trend='add', seasonal=None).fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
    return des_model.fittedvalues

formatted_df['SES_Prediction'] = apply_ses(formatted_df['Quantity'], alpha=alpha_ses)  # Apply SES
formatted_df['DES_Prediction'] = apply_des(formatted_df['Quantity'], alpha=alpha_ses, beta=beta_des)  # Apply DES

# RMSE & R2 METRICS
results = []  # Initialize list to store results
# Evaluate models for RMSE and R2
for column in ['MA_Prediction', 'Smoothed_Weighted_Prediction', 'Linear_Regression_Prediction',
               'Polynomial_Regression_Prediction_Degree_2', 'Polynomial_Regression_Prediction_Degree_3',
               'SES_Prediction', 'DES_Prediction']:
    valid_data = formatted_df.dropna(subset=['Quantity', column])  # Filter valid data
    if valid_data.empty:  # Skip if no valid data
        print(f"Skipping {column} due to lack of valid data.")
        continue
    try:
        rmse = np.sqrt(mean_squared_error(valid_data['Quantity'], valid_data[column]))  # Calculate RMSE
        r2 = r2_score(valid_data['Quantity'], valid_data[column])  # Calculate R2
        results.append({'Model': column, 'RMSE': rmse, 'R2': r2})  # Append results
    except Exception as e:
        print(f"Error evaluating {column}: {e}")  # Handle errors

results_df = pd.DataFrame(results)  # Convert results to a DataFrame

# BEST MODEL SELECTION
best_model_row = results_df.loc[results_df['R2'].idxmax()]  # Select best model by highest R2
best_model_name = best_model_row['Model']  # Best model name
best_model_r2 = best_model_row['R2']  # Best model R2 value
next_period_prediction = formatted_df.loc[formatted_df['Period'] == 13, best_model_name].values[0]  # Predict next period

# PLOT ALL GRAPHS
plt.figure(figsize=(12, 8))  # Set figure size
plt.plot(formatted_df['Period'], formatted_df['Quantity'], label='Data Historis', marker='o', color='black')  # Plot historical data
plt.plot(formatted_df['Period'], formatted_df['MA_Prediction'], label='Prediksi MA', linestyle='--', color='orange')  # Plot MA predictions
plt.plot(formatted_df['Period'], formatted_df['Smoothed_Weighted_Prediction'], label='Prediksi WMA', linestyle='--', color='blue')  # Plot WMA predictions
plt.plot(formatted_df['Period'], formatted_df['Linear_Regression_Prediction'], label='Prediksi Regresi Linear', linestyle='--', color='green')  # Plot linear regression predictions
plt.plot(formatted_df['Period'], formatted_df['Polynomial_Regression_Prediction_Degree_2'], label='Prediksi Regresi Polinomial Derajat 2', linestyle='--', color='red')  # Plot polynomial regression (degree 2) predictions
plt.plot(formatted_df['Period'], formatted_df['Polynomial_Regression_Prediction_Degree_3'], label='Prediksi Regresi Polinomial Derajat 3', linestyle='--', color='purple')  # Plot polynomial regression (degree 3) predictions
plt.plot(formatted_df['Period'], formatted_df['SES_Prediction'], label='Prediksi SES', linestyle='--', color='magenta')  # Plot SES predictions
plt.plot(formatted_df['Period'], formatted_df['DES_Prediction'], label='Prediksi DES', linestyle='--', color='yellow')  # Plot DES predictions
plt.xlabel('Period')  # Set x-axis label
plt.ylabel('Quantity')  # Set y-axis label
plt.title(f'Prediksi Quantity untuk P/N {pn_to_filter}')  # Set plot title
plt.legend()  # Add legend
plt.grid(False)  # Disable grid
plt.show()  # Display the plot

# PLOT BEST MODEL
plt.figure(figsize=(12, 8))  # Set figure size
plt.plot(formatted_df['Period'], formatted_df['Quantity'], label='Data Historis', marker='o', color='black')  # Plot historical data
plt.plot(formatted_df['Period'], formatted_df[best_model_name], label=f'Best Model: {best_model_name}', linestyle='-', color='blue')  # Plot best model predictions
plt.xlabel('Period')  # Set x-axis label
plt.ylabel('Quantity')  # Set y-axis label
plt.title(f'Prediksi menggunakan {best_model_name} untuk 1 bulan kedepan')  # Set plot title
plt.legend()  # Add legend
plt.grid(False)  # Disable grid
plt.show()  # Display the plot

# DISPLAY DATAFRAMES
print(tabulate(formatted_df, headers='keys', tablefmt='fancy_grid'))  # Print formatted data
print(tabulate(results_df.sort_values(by='R2', ascending=False), headers='keys', tablefmt='fancy_grid'))  # Print sorted results

# DISPLAY BEST MODEL SUMMARY
table_data = [
    ["P/N", pn_to_filter],
    ["Best Model", best_model_name],
    ["Best R-squared", f"{best_model_r2:.4f}"],
    ["Prediction for 13th Period", f"{next_period_prediction:.0f}"]
]
print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="fancy_grid"))  # Print summary

# EXPORT RESULTS TO JSON
output_file_path = 'results.json'  # Path to output file
formatted_df.to_json(output_file_path, orient='records', indent=4)  # Save DataFrame to JSON

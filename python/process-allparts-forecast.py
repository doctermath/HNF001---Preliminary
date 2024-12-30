# %% [markdown]
# Python Script to generate Spare Parts Demand for 13th Month by 12 Last Month  
# Original Author     : Michael Brandon  
# Original Modified   : 2024-12-24  
# Modified By         : Unknown Player  
# Modified Date       : 2024-12-24  

# %%
# PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing
from tabulate import tabulate
import os
import json
import sys
import requests
import math

# Set Display Width Longer
# pd.options.display.max_colwidth = 100 # 100 for long width

# %%
# Retrive JSON Data From API.
url = "http://172.16.1.59:18080/v1/web/get-spare-parts-all-history-data"

# Fetch JSON data from the API
response = requests.get(url)
response.raise_for_status()  # Raise an error if the request fails
df = response.json()  # Parse JSON data

# Convert JSON to Pandas DataFrame
data = pd.DataFrame(df['data'])
# display(data)

# %%
# Retrive Data from json
if 1 == 0:
    with open('from_api.json', 'r') as file:
        jsonData = json.load(file)
    jsonData = jsonData['data'][:5]
    data = pd.DataFrame(jsonData)

# %%
# Add Metric to the data

# get mean and standart deviation of first 12 data
data['mean_12'] = data['D'].apply(lambda x: np.mean(x[:12]))
data['std_12'] = data['D'].apply(lambda x: np.std(x[:12], ddof=1))

# get upper bound from mean and std
data['ub'] = data['mean_12'] + 1.5 * data['std_12']

# limit the original data to upper bound
data['clipped_d'] = data.apply(lambda row: np.clip(row['D'][:12], 0, row['ub']).tolist(), axis=1)
# display(data)

# %%
# Calculate Simple Moving Average ?? this only calculate mean of data, not SMA
def calculate_ma(list):
    oldData = []
    newData = []
    for i in list:
        # store calculated data to old list
        oldData.append(i)
        newData.append(np.mean(oldData))
    return newData

data['ma'] = data['clipped_d'].apply(calculate_ma)
data['ma_result'] = data['ma'].apply(lambda x: x[-1:])
# data['ma_result'] = data['clipped_d'].apply(lambda x: np.mean(x))
# display(data)


# %%
# Calculate Exponential Weighted Moving Average (EWMA)
def ewma(list, alpha):
    df = pd.DataFrame(list)
    df['ewma'] = df.ewm(alpha=alpha, adjust=False).mean()
    return df['ewma'].tolist()

ewma_alpha = 0.3
data['ewma'] = data['clipped_d'].apply(lambda x: ewma(x, ewma_alpha))
data['ewma_result'] = data['ewma'].apply(lambda x: x[-1:])
# display(data['ewma'][0])
# display(data)

# %%
#  Calculate Linear Regression
def lr(x):
    df = pd.DataFrame()
    df['y'] = x
    df['x'] = range(1, len(df) + 1)
    model =  LinearRegression()
    model.fit(df[['x']], df['y'])
    df.loc[len(df), 'x'] = len(df) + 1
    return model.predict(df[['x']])

data['lr'] = data['clipped_d'].apply(lambda x: lr(x))
data['lr_result'] = data['lr'].apply(lambda x: x[-1:])
# display(data)


# %%
# Calculate Polynomial Regression
def pr(x, pr_degree):
    df = pd.DataFrame()
    df['y'] = x
    df['x'] = range(1, len(df) + 1)

    X = df[['x']]  # Independent variable (reshape to 2D array)
    y = df['y']    # Dependent variable

    poly = PolynomialFeatures(degree=pr_degree)  # Create polynomial features
    X_poly = poly.fit_transform(X)  # Transform input features
    poly_model = LinearRegression()  # Initialize linear regression model
    poly_model.fit(X_poly, y)  # Fit polynomial model

    df.loc[len(df), 'x'] = len(df) + 1
    X_all_poly = poly.transform(df[['x']])
    return poly_model.predict(X_all_poly)  

data['pr2'] = data['clipped_d'].apply(lambda x: pr(x, 2))
data['pr2_result'] = data['pr2'].apply(lambda x: x[-1:])
data['pr3'] = data['clipped_d'].apply(lambda x: pr(x, 3))
data['pr3_result'] = data['pr3'].apply(lambda x: x[-1:])
# display(data)


# %%
# Calculate Single Exponential Smoothing
def ses(x, alpha):
    df = pd.DataFrame()
    df['y'] = x
    df['x'] = range(1, len(df) + 1)
    df.loc[len(df), 'x'] = len(df) + 1

    new_data = SimpleExpSmoothing(df['y']).fit(smoothing_level=alpha, optimized=False).fittedvalues
    return new_data.tolist()

data['ses'] = data['clipped_d'].apply(lambda x: ses(x, 0.8))
data['ses_result'] = data['ses'].apply(lambda x: x[-1:])
# display(data)

# %%
# Calculate Double Exponential Smoothing
def des(x, alpha, beta):
    df = pd.DataFrame()
    df['y'] = x
    df['x'] = range(1, len(df) + 1)
    df.loc[len(df), 'x'] = len(df) + 1

    new_data = ExponentialSmoothing(df['y'], trend='add', seasonal=None).fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False).fittedvalues
    return new_data.tolist()

data['des'] = data['clipped_d'].apply(lambda x: des(x, 0.8, 0.3))
data['des_result'] = data['des'].apply(lambda x: x[-1:])
# display(data)

# %%
# calculate R2 Score and RMSE
def metric(x):
    period_lenght = len(x['clipped_d'])
    df = pd.DataFrame()
    df['period'] = range(1, period_lenght + 1)
    df['qty'] = x['clipped_d'][:period_lenght]
    df['ma'] = x['ma'][:period_lenght]
    df['ewma'] = x['ewma'][:period_lenght]
    df['lr'] = x['lr'][:period_lenght]
    df['pr2'] = x['pr2'][:period_lenght]
    df['pr3'] = x['pr3'][:period_lenght]
    df['ses'] = x['ses'][:period_lenght]
    df['des'] = x['des'][:period_lenght]
    # display(df)
    
    # df = pd.concat([pd.DataFrame(x['lr']), df], axis=1)
    result = []
    for i in df.iloc[:, -7:]:
        rmse = np.sqrt(mean_squared_error(df['qty'], df[i]))  # Calculate RMSE
        r2 = r2_score(df['qty'], df[i])  # Calculate R2
        result.append({'model': i, 'RMSE': rmse, 'R2': r2})
        
    # display(result)
    # df_result = pd.DataFrame()

    
    return result


data['metric'] = data.apply(lambda x: metric(x), axis=1)

# display(data['metric'][1])
# display(data)

# %%
# BEST MODEL SELECTION
def best_select(x, key):
    return max(x, key=lambda x: x['R2'])[key]

def best_number(x):
    return x[x['best_model']][-1]

data['best_model'] = data['metric'].apply(lambda x: best_select(x, 'model'))
data['best_r2'] = data['metric'].apply(lambda x: best_select(x, 'R2'))
data['best_value'] = data.apply(lambda x: best_number(x), axis=1)
data['FD'] = round(data['best_value'])
# display(data)


# %%
# Send Data Back To API

# API endpoint
url = "http://172.16.1.59:18080/v1/web/parts-forecast-result"

data2 = data[['period', 'branch', 'agency', 'partno', 'FD', 'mean_12', 'std_12', 'ub']]
json2 = data2.to_dict(orient='records')

# Send POST request
response = requests.post(url, json=json2)

# Print response
print(f"Status Code: {response.status_code}")
print(f"Response Body: {response.text}")
print(response.json().get("success", "No status key found"))

# %%
# Convert DataFrame to JSON and write to a file

# Specify the target directory where you want to save the file
# target_folder = 'output/'  # Update this path

# # Ensure the directory exists (create if it doesn't)
# os.makedirs(target_folder, exist_ok=True)

# # Full file path
# file_path = os.path.join(target_folder, 'result.json')

# data.to_json(file_path, orient='records', lines=False)
# print(True)



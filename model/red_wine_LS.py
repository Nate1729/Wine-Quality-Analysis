""" Least Squares regression for red wine using 
residual sugar, total sulfur dioxide, pH, sulphates, and alcohol"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


features = ['residual sugar', 'total sulfur dioxide', 'pH', 'sulphates', 'alcohol']
df_red_wine = pd.read_csv('../data/winequality-red.csv', delimiter=';', usecols=features)
df_quality = pd.read_csv('../data/winequality-red.csv', delimiter=';', usecols=['quality'])

# First we should normalize the data
scaler = MinMaxScaler()
scaler.fit(df_red_wine)
df_red_wine_norm = scaler.transform(df_red_wine)

# Splitting data into training and test sets
# Supervised learning
x_train, x_test, y_train, y_test = train_test_split(df_red_wine_norm, df_quality)

# Perform least squares regression
reg = LinearRegression().fit(x_train, y_train)
print(f'Least Squares R-squared: {reg.score(x_test, y_test)}')

# Perform Ridge regression
reg_ridge = Ridge()
reg_ridge.fit(x_train, y_train)
print(f'Ridge Regression R-squared: {reg_ridge.score(x_test, y_test)}')


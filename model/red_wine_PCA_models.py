
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# Load data and separate into inputs and outputs
red_wine_feat = pd.read_csv('../data/winequality-red.csv', delimiter=';')
df_quality = red_wine_feat['quality']
df_features = red_wine_feat.drop(columns=['quality'])
del red_wine_feat

# Normalizing the data
scaler = StandardScaler()
df_features = scaler.fit_transform(df_features)

# PCA - Analysis
pca = PCA(n_components=8) # 8 gives > 90% cumulative explained variation ratio
df_features_reduce = pca.fit_transform(df_features)

x_train, x_test, y_train, y_test = train_test_split(df_features_reduce, df_quality, test_size=0.25)

## LS Regression
reg = LinearRegression().fit(x_train, y_train)
print(f'Least Squares R-squared: {reg.score(x_test, y_test)}')

## Ridge Regression
reg_ridge = Ridge().fit(x_train, y_train)
print(f'Ridge Regression R-squared: {reg_ridge.score(x_test, y_test)}')

## Random Forest Model
model = RandomForestClassifier()
model.fit(x_train, y_train)
model_prediction = model.predict(x_test)
model_accuracy = accuracy_score(model_prediction, y_test)
print(f'Forest Classification Accuracy: {model_accuracy}')

## Random Forest using SMOTE
sm = SMOTE()
print(x_train.shape)
print(y_train.shape)
x_train_resample, y_train_resample = sm.fit_resample(x_train, y_train)

model = RandomForestClassifier()
model.fit(x_train_resample, y_train_resample)
model_prediction = model.predict(x_test)
model_accuracy = accuracy_score(model_prediction, y_test)
print(f'SMOTE-adjusted Forest Classification Accuracy: {model_accuracy}')
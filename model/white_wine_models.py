import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

white_wine_data = pd.read_csv('../data/winequality-white.csv', delimiter=';')
white_wine_quality = white_wine_data['quality']
white_wine_feat = white_wine_data.drop(columns=['quality'])
del white_wine_data

# Normalizing the data
scaler = StandardScaler()
white_wine_feat = scaler.fit_transform(white_wine_feat)

# PCA Transform
# We do this because the features are highly correlated
pca = PCA()
white_wine_feat_pca = pca.fit_transform(white_wine_feat)

# Splitting the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(white_wine_feat_pca, white_wine_quality, test_size=0.25)

# Linear regression
reg = LinearRegression().fit(x_train, y_train)
print(f'Least Squares R-Squared: {reg.score(x_test, y_test)}')

# Random forest model
# No SMOTE since it didn't work for red wine
model = RandomForestClassifier()
model.fit(x_train, y_train)
model_prediction = model.predict(x_test)
model_accuracy = accuracy_score(model_prediction, y_test)
print(f'Random Forest Accuracy: {model_accuracy}')
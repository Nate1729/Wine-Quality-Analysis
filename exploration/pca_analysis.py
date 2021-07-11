import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing 
import matplotlib.pyplot as plt

df_red_wine = pd.read_csv('../data/winequality-red.csv', delimiter=';').drop(columns=['quality'])

# We have to normalize the data

#for col in list(df_red_wine.columns):
#	df_red_wine[col] = scaler.fit_transform(df_red_wine[col])
df_normalized = preprocessing.scale(df_red_wine, axis=0)

pca = PCA() # Not specifying n_components because I want all of them
pca.fit(df_normalized)

plt.plot(pca.explained_variance_ratio_)
plt.show()
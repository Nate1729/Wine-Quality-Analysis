import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler

# This script plots all of the attributes of the data set as scatter 
# plots to determine if any of them depend on each other

def combination_scatter(df, folder):
	"""
	This function plots every combination of columns on a 
	scatter plot using matplotlib. The data must be numeric
	Inputs
	------
	df : DataFrame
		DataFrame containing all the independent variables to be analyzed"""
	col = combinations(df.columns, 2) # Only want combinations

	for label in col:
		plt.scatter(df[label[0]], df[label[1]])
		plt.xlabel(label[0])
		plt.ylabel(label[1])
		plt.savefig(f'./{folder}/{label[1]}_vs_{label[0]}.png', dpi=800)
		plt.clf()

def cov_heatmap(df, title='heatmap', axis_tilt=0, kill_diag=True):
	cov = df.cov()	
	labels = cov.columns

	fig, ax = plt.subplots()
	im = ax.imshow(cov)

	# Configure axes
	ax.set_xticks(np.arange(len(labels)))
	ax.set_yticks(np.arange(len(labels)))
	# Label Axes
	ax.set_xticklabels(labels)
	plt.xticks(rotation=axis_tilt)
	ax.set_yticklabels(labels)
	# Create color bar
	cbar = ax.figure.colorbar(im, ax=ax)
	# Set title
	plt.title(title)
	# Full size figure
	fig.tight_layout()
	plt.savefig(f'{title}.png', dpi=800)

def normalize_dataframe(df, center=False):
	""" 
	This divides each series by it standard deviation.
	It does *not* center the data at zero"""

	if center:
		for col in list(df.columns):
			df[col] = (df[col] - df[col].mean())/ df[col].std()
	else:
		for col in list(df.columns):
			df[col] = df[col]/df[col].std()

	return df

def feature_box_plot(df_data, output, location='./'):
	col_feat = list(df_data.columns)
	col_feat.remove(output)

	for col in col_feat:
		sns.boxplot(x=output, y=col, data=df_data)
		plt.savefig(f'{location}_{col}_vs_{output}.png', dpi=800)
		plt.clf()


# Loading data
df_red_wine = pd.read_csv('../data/winequality-red.csv', delimiter=';')
df_white_wine = pd.read_csv('../data/winequality-white.csv', delimiter=';')

# Make red wine box plots
#feature_box_plot(df_red_wine, 'quality', 'box_plot_red/red_')

# Normalize the data
# df_red_wine = normalize_dataframe(df_red_wine)
# df_white_wine = normalize_dataframe(df_white_wine)

# Plotting Covariance matrix
#cov_heatmap(df_red_wine.drop(columns=['density', 'free sulfur dioxide', 'citric acid', 'fixed acidity', 'chlorides', 'volatile acidity', 'quality']), "RedWineCovariance_adj", 90)
#cov_heatmap(df_white_wine, "WhiteWineCovariance", 90)


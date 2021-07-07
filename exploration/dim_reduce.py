import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

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

# Loading data
df_red_wine = pd.read_csv('../data/winequality-red.csv', delimiter=';')
df_white_wine = pd.read_csv('../data/winequality-white.csv', delimiter=';')

combination_scatter(df_white_wine.drop(columns=['quality']), 'white_wine_scatter')
combination_scatter(df_red_wine.drop(columns=['quality']), 'red_wine_scatter')
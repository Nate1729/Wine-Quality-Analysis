""" This is a bar char of wine qualilty ratings"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_data = pd.read_csv('../data/winequality-red.csv', delimiter=';')
sns.countplot(x='quality', data=df_data)
plt.savefig('red_wine_countplot.png', dpi=800)
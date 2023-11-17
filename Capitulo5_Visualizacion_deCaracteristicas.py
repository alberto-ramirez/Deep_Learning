import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

df = pd.read_csv("Precios_casas.csv")
cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'price']
sb.pairplot( df[cols], size=2.5 )
plt.tight_layout()
plt.show()

cm = np.corrcoef( df[cols].values.T )
sb.set( font_scale=1.5 )
hm = sb.heatmap( cm, cbar=True, annot=True, square=True, fmt='.2F', annot_kws={'size':15}, yticklabels = cols, xticklabels = cols )
plt.show()

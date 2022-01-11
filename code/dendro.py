# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 09:57:27 2021

@author: katha
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
import pandas as pd

import seaborn as sns
import matplotlib.colors as colors

from matplotlib import rc
rc('text', usetex=True)

colors_test = sns.cubehelix_palette(start=-1, rot=-1, light=.75, dark=0.4,n_colors=6)[::-1]  
colors_as_hex = [colors.rgb2hex(c) for c in colors_test]
df = pd.read_csv('../data/data_KN.csv', sep=";").fillna(value=-1)
df = df.loc[:,df.apply(pd.Series.nunique) != 1]
X = df.drop(df.iloc[:, 0:10], axis=1)

X2 = df.drop(df.iloc[:, 0:10], axis=1)
fig, (ax1) = plt.subplots(1, figsize=(12, 3))

# adapted from https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
corr = spearmanr(X2).correlation
corr_linkage = hierarchy.ward(corr)
 
hierarchy.set_link_color_palette(colors_as_hex)
dendro = hierarchy.dendrogram(
    corr_linkage, ax=ax1,color_threshold=5,above_threshold_color='dimgray', no_labels=True
)
plt.xlabel('Features',fontsize=14)
plt.ylabel('Distance',fontsize=14)
plt.yticks(fontsize=14)
dendro_idx = np.arange(0, len(dendro['ivl']))
 
ax1.set_yscale('log',base=5)
ax1.set_ylim(1,125)

fig.tight_layout()
plt.plot([0, 5000], [5,5], color = 'crimson', linestyle=":")
plt.savefig("dendro.pdf",bbox_inches="tight",pad_inches=0.3)
plt.show()


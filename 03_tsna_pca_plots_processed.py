# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

X_train = pd.read_csv("X_train_processed.csv")
y_train = pd.read_csv('y_train.csv', header = None, index_col = 0, squeeze = bool)
# -

X_train_processed = X_train.drop('Unnamed: 0', axis=1)

X_train

y_train.shape

# +
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.pyplot import figure
figure(num=None, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')

tsne = TSNE(n_components=2, random_state=0)
transformed_data = tsne.fit_transform(X_train_processed[:500])
k = np.array(transformed_data)

colors = ['red', 'green']

plt.scatter(k[:, 0], k[:, 1], c=y_train[:500], zorder=10, s=2, cmap=matplotlib.colors.ListedColormap(colors))
# -

from sklearn.decomposition import PCA 
pca = PCA(n_components = 2)
X2D = pca.fit_transform(X_train_processed)

pca.explained_variance_ratio_

figure(num=None, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
colors = ['red', 'green']
plt.scatter(X2D[:, 0], X2D[:, 1], c=y_train, zorder=10, s=2, cmap=matplotlib.colors.ListedColormap(colors))

pca = PCA()
pca.fit(X_train_processed)
cumsum = np.cumsum(pca.explained_variance_ratio_) 
d = np.argmax(cumsum >= 0.95) + 1

d

pca = PCA(n_components=0.95) 
X_reduced = pca.fit_transform(X_train_processed)





#Import libraries:

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Load and process data:

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
target = iris.target
print('real dataset is:')
print(data)

# Standardize data for better results
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

#Perform SVD:

# Use SVD from Scikit-learn for simplicity
pca = PCA(n_components=2)  # Reduce to 2 dimensions
pca.fit(data_scaled)

# Get transformed data and singular values
data_pca = pca.transform(data_scaled)
singular_values = pca.singular_values_
print('transformed dataset is:')
print(data_pca)

#Visualization:

import matplotlib.pyplot as plt

# Use different markers for different species
print('different markers for different species')
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=target, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Iris data projected onto first two principal components')
plt.show()

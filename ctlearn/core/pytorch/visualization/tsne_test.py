import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create a TSNE instance with desired parameters
tsne = TSNE(n_components=2, random_state=42)

# Perform TSNE
X_embedded = tsne.fit_transform(X)

# Plot the embedded points with matplotlib
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green']
for i, color in enumerate(colors):
    plt.scatter(X_embedded[y == i, 0], X_embedded[y == i, 1], c=color, label=iris.target_names[i])

plt.legend(loc='best')
plt.title('TSNE visualization of the Iris dataset')
plt.xlabel('TSNE 1')
plt.ylabel('TSNE 2')
plt.show()
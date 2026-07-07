from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

# Number of data points
n = int(input("Enter number of data points: "))

data = []

# Input data points
for i in range(n):
    x = float(input(f"Enter x value for point {i + 1}: "))
    y = float(input(f"Enter y value for point {i + 1}: "))
    data.append([x, y])

# Number of clusters
k = int(input("Enter number of clusters: "))

# Convert to NumPy array
X = np.array(data)

# Perform Agglomerative Clustering
Z = linkage(X, method='ward')

# Get cluster labels
labels = fcluster(Z, k, criterion='maxclust')

# Output cluster labels
print("\nCluster Labels:")
for i in range(n):
    print(f"Point {data[i]} -> Cluster {labels[i]}")
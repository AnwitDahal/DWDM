from sklearn.cluster import KMeans
import numpy as np

# Number of data points
n = int(input("Enter number of data points: "))

data = []

# Input data points
for i in range(n):
    x = float(input(f"Enter x-coordinate of point {i + 1}: "))
    y = float(input(f"Enter y-coordinate of point {i + 1}: "))
    data.append([x, y])

# Number of clusters
k = int(input("Enter number of clusters (K): "))

# Convert to NumPy array
X = np.array(data)

# Create and train K-Means model
kmeans = KMeans(n_clusters=k, random_state=0)

# Fit the model
kmeans.fit(X)

# Output cluster labels
print("\nCluster Labels:")
print(kmeans.labels_)

# Output centroids
print("\nCentroids:")
print(kmeans.cluster_centers_)
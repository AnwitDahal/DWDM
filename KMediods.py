from pyclustering.cluster.kmedoids import kmedoids
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
k = int(input("Enter number of clusters (K): "))

# Initial medoid indexes
initial_medoids = list(range(k))

# Create K-Medoids model
kmedoids_instance = kmedoids(data, initial_medoids)

# Train model
kmedoids_instance.process()

# Get clusters and medoids
clusters = kmedoids_instance.get_clusters()
medoids = kmedoids_instance.get_medoids()

# Output clusters
print("\nClusters:")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}: {cluster}")

# Output medoid points
print("\nMedoid Points:")
for medoid in medoids:
    print(data[medoid])
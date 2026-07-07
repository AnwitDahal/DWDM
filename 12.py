import numpy as np

# Distance function (Euclidean)
def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


# DBSCAN algorithm
def dbscan(X, eps, min_pts):
    n = len(X)
    labels = [-1] * n      # -1 means noise
    visited = [False] * n
    cluster_id = 0

    def region_query(idx):
        neighbors = []
        for i in range(n):
            if distance(X[idx], X[i]) <= eps:
                neighbors.append(i)
        return neighbors

    def expand_cluster(idx, neighbors):
        labels[idx] = cluster_id
        i = 0

        while i < len(neighbors):
            point = neighbors[i]

            if not visited[point]:
                visited[point] = True
                new_neighbors = region_query(point)

                if len(new_neighbors) >= min_pts:
                    for neighbor in new_neighbors:
                        if neighbor not in neighbors:
                            neighbors.append(neighbor)

            if labels[point] == -1:
                labels[point] = cluster_id

            i += 1

    for i in range(n):
        if visited[i]:
            continue

        visited[i] = True
        neighbors = region_query(i)

        if len(neighbors) < min_pts:
            labels[i] = -1
        else:
            expand_cluster(i, neighbors)
            cluster_id += 1

    return labels


# ---------------- USER INPUT ----------------

n = int(input("Enter number of data points: "))

data = []

for i in range(n):
    x = float(input(f"Enter x value for point {i + 1}: "))
    y = float(input(f"Enter y value for point {i + 1}: "))
    data.append([x, y])

eps = float(input("Enter eps: "))
min_pts = int(input("Enter min_points: "))

X = np.array(data)

labels = dbscan(X, eps, min_pts)

print("\nCluster Labels:")
for i in range(n):
    print(f"Point {data[i]} -> Cluster {labels[i]}")
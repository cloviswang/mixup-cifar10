import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def gen_clusters():
    mean1 = [0, 0]
    cov1 = [[1, 0], [0, 10]]
    data = np.random.multivariate_normal(mean1, cov1, 100)

    mean2 = [10, 10]
    cov2 = [[10, 0], [0, 1]]
    data = np.append(data,
                     np.random.multivariate_normal(mean2, cov2, 100),
                     0)

    mean3 = [10, 0]
    cov3 = [[3, 0], [0, 4]]
    data = np.append(data,
                     np.random.multivariate_normal(mean3, cov3, 100),
                     0)

    return np.round(data, 4)


def show_scatter(data, colors):
    x, y = data.T
    plt.scatter(x, y, c=colors)
    plt.axis()
    plt.title("scatter")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


data = gen_clusters()
show_scatter(data, 'k')

# 初始化参数为 k-means++
estimator = KMeans(init='k-means++', n_clusters=3, n_init=3)
estimator.fit(data)
label2color = ['r', 'g', 'b']
colors = [label2color[i] for i in estimator.labels_]
show_scatter(data, colors)

centroids = estimator.cluster_centers_
print(centroids)

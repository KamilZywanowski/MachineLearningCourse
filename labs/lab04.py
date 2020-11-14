from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics, cluster, decomposition, datasets, manifold
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import numpy as np
import pandas as pd
import pickle
import gap_statistic


def task1():
    X, y = datasets.load_iris(return_X_y=True, as_frame=False)

    # scatter 3 features
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 2], X[:, 3], c=y, marker='o')

    ax.set_xlabel('0')
    ax.set_ylabel('2')
    ax.set_zlabel('3')
    #
    # plt.show()
    # choose clustering KMeans/MeanShift/AffinityPropagation/Agglomerative etc
    # clustering = cluster.KMeans(3)
    clustering = cluster.KMeans()
    clustering.fit(X)
    # print(clustering.cluster_centers_)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 2], X[:, 3], c=clustering.labels_, marker='o')
    # for x in clustering.cluster_centers_:
    #     ax.scatter(x[0], x[2], x[3], c='Red', marker='*')
    ax.set_xlabel('0')
    ax.set_ylabel('2')
    ax.set_zlabel('3')

    clustering_quality_gt = metrics.adjusted_rand_score(y, clustering.labels_)
    clustering_quality_no_gt = metrics.calinski_harabasz_score(X, clustering.labels_)
    print(clustering_quality_gt, clustering_quality_no_gt)

    plt.show()
    # print(kmeans.labels_)
    # print(y)


def task2():
    X, y = datasets.load_iris(return_X_y=True, as_frame=False)

    # choose clustering KMeans/MeanShift/AffinityPropagation/Agglomerative etc
    # clustering = cluster.KMeans(3)
    inertias = {}
    for clusters in range(2, 16):
        clustering = cluster.KMeans(clusters)
        clustering.fit(X)
        inertias[clusters] = clustering.inertia_
        # print(clustering.cluster_centers_)
    plt.figure()
    plt.plot(inertias.keys(), inertias.values())

    plt.figure()
    optimalK = gap_statistic.OptimalK(n_jobs=2, parallel_backend='joblib')
    n_clusters = optimalK(X, cluster_array=np.arange(1, 15))
    optimalK.plot_results()
    plt.show()


def task3():
    X, y = datasets.load_digits(return_X_y=True, as_frame=False)

    pca = decomposition.PCA(n_components=2)
    sne = manifold.TSNE(n_components=2)

    X_pca = pca.fit_transform(X)
    X_sne = sne.fit_transform(X)

    plt.figure("No reduction (no sense with digits, use with iris)")
    plt.scatter(X[:, 0], X[:, 1], c='Green')
    plt.figure("PCA")
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c='Red')
    plt.figure("TSNE")
    plt.scatter(X_sne[:, 0], X_sne[:, 1], c='Blue')
    plt.show()

def main():
    # task1()
    # task2()
    task3()


if __name__ == '__main__':
    main()

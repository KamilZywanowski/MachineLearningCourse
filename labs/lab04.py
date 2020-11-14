from sklearn.model_selection import train_test_split, RandomizedSearchCV, PredefinedSplit, GridSearchCV
from sklearn import metrics, cluster, decomposition, datasets, manifold, ensemble
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import numpy as np
import pandas as pd
import pickle
import gap_statistic
import os
import cv2

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


def load_hist_features(folder):
    all_features = []
    for impath in os.listdir(folder):
        img = cv2.imread(folder + impath, cv2.IMREAD_COLOR)
        if img is None:
            print(f'Error loading image {impath}')
            continue
        # get hists
        hist_b = np.squeeze(cv2.calcHist([img], [0], None, [8], [0, 256]))
        hist_g = np.squeeze(cv2.calcHist([img], [1], None, [8], [0, 256]))
        hist_r = np.squeeze(cv2.calcHist([img], [2], None, [8], [0, 256]))
        # normalize
        features = np.concatenate((hist_b, hist_g, hist_r))
        features = features/np.sum(features)
        all_features.append(features)

    return all_features


def load_beach_forest():
    forest_hists = np.array(load_hist_features('forest/'), dtype=np.double)
    beach_hists = np.array(load_hist_features('beach/'), dtype=np.double)
    X = np.concatenate((forest_hists, beach_hists))
    y = [0] * len(forest_hists) + [1] * len(beach_hists)
    return X, y


def task4():
    X, y = load_beach_forest()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=42, stratify=y)

    # print(X)

    # visualization
    # pca = decomposition.PCA(n_components=2)
    # sne = manifold.TSNE(n_components=2)
    # X_pca = pca.fit_transform(X)
    # X_sne = sne.fit_transform(X)
    #
    # plt.figure("PCA")
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c='Red')
    # plt.figure("TSNE")
    # plt.scatter(X_sne[:, 0], X_sne[:, 1], c='Blue')
    # plt.show()

    clustering = cluster.KMeans(n_clusters=2)
    clustering.fit(X_train)

    print("Clustering: \n", clustering.predict(X_test))
    print("GT: \n", np.array(y_test))


def task5():
    X, y = load_beach_forest()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                        random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5,
                                                        random_state=42, stratify=y_test)

    param_grid = {
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'criterion': ['gini', 'entropy']
    }

    train_and_val_X = np.concatenate((X_train, X_val))
    train_and_val_y = np.concatenate((y_train, y_val))

    test_fold = np.concatenate(([-1] * len(X_train), [0] * len(X_val)))
    ps = PredefinedSplit(test_fold=test_fold)

    classifier = ensemble.RandomForestClassifier()
    grid_search = GridSearchCV(classifier, param_grid, cv=ps)
    grid_search.fit(train_and_val_X, train_and_val_y)
    print(grid_search.best_params_)
    print(grid_search.best_score_)




def main():
    # task1()
    # task2()
    # task3()
    # task4()
    task5()

if __name__ == '__main__':
    main()

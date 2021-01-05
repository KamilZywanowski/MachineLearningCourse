from sklearn.experimental import enable_iterative_imputer
from sklearn import metrics, cluster, decomposition, datasets, manifold, ensemble, svm, model_selection, impute
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import gap_statistic
import os
import cv2


def task1():
    # X, y = datasets.fetch_openml('diabetes', return_X_y=True, as_frame=True)
    # print(X)
    #
    # print(X.info())
    # print(X.describe())
    X, y = datasets.fetch_openml('diabetes', return_X_y=True, as_frame=True)
    # print(X)
    # print(y)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_2 = X_train.copy()

    plt.figure()
    # X_train.boxplot()
    X_train.hist()
    # plt.show()

    # # seaborn library
    # sns.boxplot(x=X_train['mass'])
    # plt.show()

    imputer = impute.SimpleImputer(missing_values=0.0, strategy='mean')
    # print(X[:, 5].reshape(-1, 1))
    # imputer.fit(X[:, 5].reshape(-1, 1))
    # X[:, 5] = imputer.transform(X[:, 5].reshape(-1, 1))

    X_train[['mass']] = imputer.fit_transform(X_train[['mass']])
    X_train[['skin']] = imputer.fit_transform(X_train[['skin']])

    plt.figure()
    # X_train.boxplot()
    X_train.hist()
    # plt.show()

    imputer_ii = impute.KNNImputer(missing_values=0.0, n_neighbors=2)
    X_train_2[['mass']] = imputer_ii.fit_transform(X_train[['mass']])
    X_train_2[['skin']] = imputer_ii.fit_transform(X_train[['skin']])

    plt.figure()
    # X_train.boxplot()
    X_train.hist()
    plt.show()

    isolation_forest = ensemble.IsolationForest(contamination='auto')
    isolation_forest.fit(X_train)
    y_predicted_outliers = isolation_forest.predict(X_test)
    print(y_predicted_outliers)


    clf_svm = svm.SVC(random_state=42)
    clf_svm.fit(X, y)
    y_predicted_svm = clf_svm.predict(X_test)
    print(metrics.classification_report(y_test, y_predicted_svm))

    clf_rf = ensemble.RandomForestClassifier(random_state=42)
    clf_rf.fit(X, y)
    y_predicted_rf = clf_rf.predict(X_test)
    print(metrics.classification_report(y_test, y_predicted_rf))
    importances = clf_rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_], axis=0)
    print('feature ranking: \n')
    indices = np.argsort(importances)[::-1]
    for f in range (X.shapee[1]):
        print(f'{f+1}. feature {indices[f], importances[indices[f]]}')

    plt.figure()
    plt.title('feat importances')
    plt.bar(range(X.shape[1]), importances[indices], color='r', yerr=std[indices], align='center')
    plt.xticks(range(X.shape[1]), indices)
    plt.show()
    # y[y == 'tested_positive'] = 1
    # y[y == 'tested_negative'] = 0
    # print(y)
    #  do result of 3 classifiers with default params of raw data

def main():
    task1()


if __name__ == '__main__':
    main()

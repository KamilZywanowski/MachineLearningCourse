from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import numpy as np
import pandas as pd


def task1(used_features):  # ONLY FIRST TWO COLUMNS, VISUALIZATION
    iris = datasets.load_iris(as_frame=True)
    # print(iris.data.describe())
    # print(iris.target)
    # print(iris.feature_names)
    # print(iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=0.8,
                                                        random_state=42, stratify=iris.target)
    # print(X_train.describe())
    # Count number of occurences
    # unique, counts = np.unique(y_test, return_counts=True)
    # print(unique, counts)
    # Wizualizujemy tylko dwie pierwsze cechy – aby móc je przedstawić bez problemu w 2D.
    # plt.scatter(X_train.loc[:, 'sepal length (cm)'], X_train.loc[:, 'sepal width (cm)'])
    # plt.axvline(x=0)
    # plt.axhline(y=0)
    # plt.title('Iris sepal features')
    # plt.xlabel('sepal length (cm)')
    # plt.ylabel('sepal width (cm)')

    scaler_mm = MinMaxScaler()
    scaler_mm.fit(X_train)
    # SCALER RETURNS NUMPY ARRAYS
    X_train_minmax_scaler = pd.DataFrame(scaler_mm.transform(X_train),
                                         columns=['sepal length (cm)', 'sepal width (cm)',
                                                  'petal length (cm)', 'petal width (cm)'])
    # plt.scatter(X_train_minmax_scaler.loc[:, 'sepal length (cm)'],
    #             X_train_minmax_scaler.loc[:, 'sepal width (cm)'])

    # scaler_standard = StandardScaler()
    # scaler_standard.fit(X_train)
    # X_train_standard_scaler = pd.DataFrame(scaler_standard.transform(X_train),
    #                                           columns=['sepal length (cm)', 'sepal width (cm)',
    #                                                    'petal length (cm)', 'petal width (cm)'])
    # plt.scatter(X_train_standard_scaler.loc[:, 'sepal length (cm)'],
    #             X_train_standard_scaler.loc[:, 'sepal width (cm)'])
    #
    # plt.show()

    X_test_minmax_scaler = pd.DataFrame(scaler_mm.transform(X_test),
                                        columns=['sepal length (cm)', 'sepal width (cm)',
                                                 'petal length (cm)', 'petal width (cm)'])
    results = {}

    svm = SVC()
    svm.fit(X_train_minmax_scaler.loc[:, used_features], y_train)
    y_predicted_svm = svm.predict(X_test_minmax_scaler.loc[:, used_features])
    results['SVM'] = classification_report(y_test, y_predicted_svm)

    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train_minmax_scaler.loc[:, used_features], y_train)
    y_predicted_logistic_regression = logistic_regression.predict(
        X_test_minmax_scaler.loc[:, used_features])
    results['Logistic Regression'] = classification_report(y_test, y_predicted_logistic_regression)

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train_minmax_scaler.loc[:, used_features], y_train)
    y_predicted_decision_tree = decision_tree.predict(
        X_test_minmax_scaler.loc[:, used_features])
    results['Decision Tree'] = classification_report(y_test, y_predicted_decision_tree)

    random_forest = RandomForestClassifier()
    random_forest.fit(X_train_minmax_scaler.loc[:, used_features], y_train)
    y_predicted_random_forest = random_forest.predict(
        X_test_minmax_scaler.loc[:, used_features])
    results['Random Forest'] = classification_report(y_test, y_predicted_random_forest)

    for key, value in results.items():
        print(f"{key} clasification report: \n{value}")

    # Plotting decision regions, mlxtend requires numpy arrays
    X_train_minmax_scaler_np = X_train_minmax_scaler.to_numpy()
    y_train_np = y_train.to_numpy()

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2)

    labels = ['SVM', 'Logistic Regression', 'Decision Tree', 'Random Forest']
    for clf, lab, grd in zip([svm, logistic_regression, decision_tree, random_forest],
                             labels,
                             itertools.product([0, 1], repeat=2)):
        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(X_train_minmax_scaler_np[:, :2], y_train_np, clf=clf, legend=2)
        plt.title(lab)
        plt.xlabel('sepal length [cm]')
        plt.ylabel('sepal width (cm)')

    plt.show()


def task2():     # ALL FEATURES, NO VISUALIZATION
    iris = datasets.load_iris(as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=0.8,
                                                        random_state=42, stratify=iris.target)
    scaler_mm = MinMaxScaler()
    scaler_mm.fit(X_train)
    # SCALER RETURNS NUMPY ARRAYS
    X_train_minmax_scaler = pd.DataFrame(scaler_mm.transform(X_train),
                                         columns=['sepal length (cm)', 'sepal width (cm)',
                                                  'petal length (cm)', 'petal width (cm)'])

    X_test_minmax_scaler = pd.DataFrame(scaler_mm.transform(X_test),
                                        columns=['sepal length (cm)', 'sepal width (cm)',
                                                 'petal length (cm)', 'petal width (cm)'])
    results = {}

    svm = SVC()
    svm.fit(X_train_minmax_scaler.loc[:, ['sepal length (cm)', 'sepal width (cm)',
                                          'petal length (cm)', 'petal width (cm)']], y_train)
    y_predicted_svm = svm.predict(X_test_minmax_scaler.loc[:, ['sepal length (cm)', 'sepal width (cm)',
                                                               'petal length (cm)', 'petal width (cm)']])
    results['SVM'] = classification_report(y_test, y_predicted_svm)

    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train_minmax_scaler.loc[:, ['sepal length (cm)', 'sepal width (cm)',
                                                          'petal length (cm)', 'petal width (cm)']], y_train)
    y_predicted_logistic_regression = logistic_regression.predict(
        X_test_minmax_scaler.loc[:, ['sepal length (cm)', 'sepal width (cm)',
                                     'petal length (cm)', 'petal width (cm)']])
    results['Logistic Regression'] = classification_report(y_test, y_predicted_logistic_regression)

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train_minmax_scaler.loc[:, ['sepal length (cm)', 'sepal width (cm)',
                                                    'petal length (cm)', 'petal width (cm)']], y_train)
    y_predicted_decision_tree = decision_tree.predict(
        X_test_minmax_scaler.loc[:, ['sepal length (cm)', 'sepal width (cm)',
                                     'petal length (cm)', 'petal width (cm)']])
    results['Decision Tree'] = classification_report(y_test, y_predicted_decision_tree)

    random_forest = RandomForestClassifier()
    random_forest.fit(X_train_minmax_scaler.loc[:, ['sepal length (cm)', 'sepal width (cm)',
                                                    'petal length (cm)', 'petal width (cm)']], y_train)
    y_predicted_random_forest = random_forest.predict(
        X_test_minmax_scaler.loc[:, ['sepal length (cm)', 'sepal width (cm)',
                                     'petal length (cm)', 'petal width (cm)']])
    results['Random Forest'] = classification_report(y_test, y_predicted_random_forest)

    for key, value in results.items():
        print(f"{key} clasification report: \n{value}")


def task3():
    pass


def task4():
    pass


def main():
    print('sepal length (cm), sepal width (cm)')
    task1(['sepal length (cm)', 'sepal width (cm)'])
    print('sepal length (cm), petal length (cm)')
    task1(['sepal length (cm)', 'petal length (cm)'])
    print("all features")
    task2()
    # task3()
    # task4()


if __name__ == '__main__':
    main()

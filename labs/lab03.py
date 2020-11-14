from sklearn import datasets
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import uniform, randint
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import numpy as np
import pandas as pd
import pickle


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
    # np.bincount()
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
    results['SVM'] = accuracy_score(y_test, y_predicted_svm)

    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train_minmax_scaler.loc[:, used_features], y_train)
    y_predicted_logistic_regression = logistic_regression.predict(
        X_test_minmax_scaler.loc[:, used_features])
    results['Logistic Regression'] = accuracy_score(y_test, y_predicted_logistic_regression)

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train_minmax_scaler.loc[:, used_features], y_train)
    y_predicted_decision_tree = decision_tree.predict(
        X_test_minmax_scaler.loc[:, used_features])
    results['Decision Tree'] = accuracy_score(y_test, y_predicted_decision_tree)

    random_forest = RandomForestClassifier()
    random_forest.fit(X_train_minmax_scaler.loc[:, used_features], y_train)
    y_predicted_random_forest = random_forest.predict(
        X_test_minmax_scaler.loc[:, used_features])
    results['Random Forest'] = accuracy_score(y_test, y_predicted_random_forest)

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


def task2():  # ALL FEATURES, NO VISUALIZATION
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
    results['SVM'] = accuracy_score(y_test, y_predicted_svm)

    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train_minmax_scaler.loc[:, ['sepal length (cm)', 'sepal width (cm)',
                                                          'petal length (cm)', 'petal width (cm)']], y_train)
    y_predicted_logistic_regression = logistic_regression.predict(
        X_test_minmax_scaler.loc[:, ['sepal length (cm)', 'sepal width (cm)',
                                     'petal length (cm)', 'petal width (cm)']])
    results['Logistic Regression'] = accuracy_score(y_test, y_predicted_logistic_regression)

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train_minmax_scaler.loc[:, ['sepal length (cm)', 'sepal width (cm)',
                                                    'petal length (cm)', 'petal width (cm)']], y_train)
    y_predicted_decision_tree = decision_tree.predict(
        X_test_minmax_scaler.loc[:, ['sepal length (cm)', 'sepal width (cm)',
                                     'petal length (cm)', 'petal width (cm)']])
    results['Decision Tree'] = accuracy_score(y_test, y_predicted_decision_tree)

    random_forest = RandomForestClassifier()
    random_forest.fit(X_train_minmax_scaler.loc[:, ['sepal length (cm)', 'sepal width (cm)',
                                                    'petal length (cm)', 'petal width (cm)']], y_train)
    y_predicted_random_forest = random_forest.predict(
        X_test_minmax_scaler.loc[:, ['sepal length (cm)', 'sepal width (cm)',
                                     'petal length (cm)', 'petal width (cm)']])
    results['Random Forest'] = accuracy_score(y_test, y_predicted_random_forest)

    for key, value in results.items():
        print(f"{key} clasification report: \n{value}")


def task3():
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
    print(svm.get_params())
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 2, 5, 10]}
    clf = GridSearchCV(svm, parameters)
    clf.fit(X_train_minmax_scaler, y_train)
    print(clf.best_params_)


    # Wersja z labow:
    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]
    clf_gs = GridSearchCV(estimator=SVC(), param_grid=param_grid, n_jobs=3, verbose=20)
    clf_gs.fit(X_train_minmax_scaler, y_train)
    print(clf_gs.cv_results_)


    decision_tree = DecisionTreeClassifier()
    print(decision_tree.get_params())
    param_dist = {"max_depth": [3, None],
                  "ccp_alpha": uniform,
                  "max_features": randint(1, 4),
                  "min_samples_leaf": randint(1, 15),
                  "criterion": ["gini", "entropy"]}
    clf = RandomizedSearchCV(decision_tree, param_dist, random_state=66)
    search = clf.fit(X_train_minmax_scaler, y_train)
    # decision_tree.set_params(search.best_params_)
    print(decision_tree.get_params())
    # print(search.best_params_)
    # for key, val in sorted(search.cv_results_.items()):
    #     print(key, val)

    clf_predicted = clf.predict(X_test_minmax_scaler.loc[:, ['sepal length (cm)', 'sepal width (cm)',
                                                             'petal length (cm)', 'petal width (cm)']])
    print("original: ", accuracy_score(y_test, clf_predicted))
    saved_model = pickle.dumps(clf)
    clf2 = pickle.loads(saved_model)
    clf2_predicted = clf2.predict(X_test_minmax_scaler.loc[:, ['sepal length (cm)', 'sepal width (cm)',
                                                               'petal length (cm)', 'petal width (cm)']])
    print("loaded from save: ", accuracy_score(y_test, clf2_predicted))


def task4():
    # mnist = datasets.fetch_openml(data_id=40996, as_frame=True)
    # mnist.data.to_csv('mnist_data.csv')
    # mnist.target.to_csv('mnist_targets.csv')
    # print(mnist.DESCR)
    mnist_data = pd.read_csv('mnist_data.csv')
    mnist_targets = pd.read_csv('mnist_targets.csv')
    # print(mnist_data)
    print(mnist_targets)

    X_train, X_test, y_train, y_test = train_test_split(mnist_data, mnist_targets.iloc[:, 1], train_size=0.8,
                                                        random_state=42, stratify=mnist_targets.iloc[:, 1])

    dtree = DecisionTreeClassifier()
    print(dtree.get_params())

    dtree.fit(X_train, y_train)
    predicted = dtree.predict(X_test)
    print(dtree.get_params())

    print("without hyperparam optimization: ", accuracy_score(y_test, predicted))

    use_snap = True
    if not use_snap:
        decision_tree = DecisionTreeClassifier()
        print(decision_tree.get_params())
        param_dist = {"max_depth": [3, None],
                      "min_samples_leaf": randint(1, 9),
                      "criterion": ["gini", "entropy"]}
        clf = RandomizedSearchCV(decision_tree, param_dist, random_state=66, n_jobs=2)
        best_model = clf.fit(X_train, y_train)
        print(best_model.best_params_)
        pickle.dump(clf, open("best_decision_tree.p", "wb"))
    else:
        best_model = pickle.load(open("best_decision_tree.p", "rb"))
        predicted = best_model.predict(X_test)
        print("loaded from save score: ", accuracy_score(y_test, predicted))



def main():
    # print('sepal length (cm), sepal width (cm)')
    # task1(['sepal length (cm)', 'sepal width (cm)'])
    # print('sepal length (cm), petal length (cm)')
    # task1(['sepal length (cm)', 'petal length (cm)'])
    # print("all features")
    # task2()
    # task3()
    task4()
    # Todo: task5() voting/stacking classifier to join models (yt)


if __name__ == '__main__':
    main()

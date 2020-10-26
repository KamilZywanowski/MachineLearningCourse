from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


def task1():
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    y = [0, 1, 1, 1]

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    print(clf.predict(
        [
            [5, 0],
            [2, 2]
        ]
    ))

    tree.plot_tree(clf, feature_names=["X1", "X2"], filled=True, rounded=True)
    plt.show()


def task2():
    brand = {"VW": 0, "Ford": 1, "Opel": 2}

    # marka, przebieg, uszkodzony
    X = [["VW", 10000, 0],
         ["VW", 10000, 1],
         ["Ford", 180000, 1],
         ["VW", 18, 0],
         ["Opel", 10000, 0],
         ["Ford", 45000, 1],
         ["VW", 18000, 1],
         ["Opel", 120000, 0]]

    for x in X:
        x[0] = brand[x[0]]

    y = [
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        1
    ]

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    print(clf.predict([[brand["VW"], 5000, 0], [brand["Ford"], 500000, 0], [brand["Opel"], 50000, 1]]))

    tree.plot_tree(clf, feature_names=["marka", "przebieg", "uszkodzony"], class_names=["nie kupić", "kupić"], filled=True, rounded=True)
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # marka = [m for m, p, u in X]
    # przebieg = [p for m, p, u in X]
    # uszk = [u for m, p, u in X]
    #
    # ax.scatter(marka, przebieg, uszk, c=y, marker='o')
    #
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    #
    # plt.show()


def task3():
    digits = datasets.load_digits()

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        digits.data, digits.target, test_size=0.33, random_state=42)


    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)

    print(f"Classification report: \n{classification_report(y_test, y_predicted)}")
    print(f"Confusion matrix: \n{confusion_matrix(y_test, y_predicted)}")

    plot_confusion_matrix(clf, X_test, y_test)
    plt.show()

    for digit, gt, pred in zip(X_test, y_test, y_predicted):
        if gt != pred and (gt == 3 or gt == 8):
            print(f"Sample {digit} classified as {pred} while it should be {gt}")
            plt.imshow(digit.reshape(8, 8), cmap=plt.cm.gray_r)
            plt.show()


def print_regressor_score(y_test: np.ndarray, y_predicted: np.ndarray) -> None:
    print(f'mae: {metrics.mean_absolute_error(y_test, y_predicted)}')
    print(f'mse: {metrics.mean_squared_error(y_test, y_predicted)}')
    print(f'r2: {metrics.r2_score(y_test, y_predicted)}')


def task4():
    data = np.loadtxt(fname='./trainingdata.txt', delimiter=",")

    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    decision_tree_regressor = tree.DecisionTreeRegressor()
    decision_tree_regressor.fit(X_train, y_train)
    y_predicted_decision_tree = decision_tree_regressor.predict(X_test)

    linear_model_regressor = LinearRegression()
    linear_model_regressor.fit(X_train, y_train)
    y_predicted_linear = linear_model_regressor.predict(X_test)

    polynomial_regressor = pipeline.Pipeline(
        [
            ('poly', preprocessing.PolynomialFeatures(degree=3)),
            ('linear', LinearRegression(fit_intercept=False))

        ]
    )

    polynomial_regressor.fit(X_train, y_train)
    y_predicted_polynomial = polynomial_regressor.predict(X_test)

    print("Decision tree:")
    print_regressor_score(y_test, y_predicted_decision_tree)
    print("Linear regressor:")
    print_regressor_score(y_test, y_predicted_linear)
    print(f'coeff: {linear_model_regressor.coef_}')
    print("Polynomial regressor:")
    print_regressor_score(y_test, y_predicted_polynomial)

    plt.scatter(X_test, y_test, c='green', marker='o')
    plt.scatter(X_test, y_predicted_decision_tree, c='red', marker='*')
    plt.scatter(X_test, y_predicted_linear, c='orange', marker='+')
    plt.scatter(X_test, y_predicted_polynomial, c='blue', marker='^')

    plt.plot(X_test, y_predicted_linear, c='orange')

    plt.show()


def main():
    # task1()
    # task2()
    # task3()
    task4()


if __name__ == '__main__':
    main()

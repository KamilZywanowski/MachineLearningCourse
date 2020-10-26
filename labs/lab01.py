from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# digits = datasets.load_digits()
# print(digits.DESCR)
# print(digits.data)
# print("x"*50)
# print(digits.images)
# plt.imshow(digits.images[0])
# , digits.target, digits.target_names, digits.images)


def todo4():
    faces = datasets.fetch_olivetti_faces()
    # print(faces)

    # print(faces.DESCR)
    # print(faces.data)
    # print(faces.target)

    x, y = datasets.fetch_olivetti_faces(return_X_y=True)

    print()


def todo5():
    from sklearn.datasets import load_boston
    boston = load_boston()
    print(boston.DESCR)
    print(boston.data)
    print(boston.target)
    print(boston['feature_names'])


def todo6():
    x, y = datasets.make_classification(
        n_samples=100,
        n_features=3,
        n_informative=3, n_redundant=0, n_repeated=0,
        n_classes=4,
        n_clusters_per_class=1,
        class_sep=5.0
    )
    # print(x)
    # print(x[:], 0)
    # plt.scatter(x[:, 0], x[:, 1], c=y)
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def todo7():
    d = datasets.fetch_openml(data_id=40536, as_frame=True)
    print(type(d))


def regressor_9000(x: float) ->float:
    if x >= 4.0:
        return 8.0
    else:
        return x * 2


def todo8():
    data = np.loadtxt('./trainingdata.txt', delimiter=",")
    print(data)

    x = data[:, 0]
    y = data[:, 1]
    y_predicted = []
    for element in x:
        y_predicted.append(regressor_9000(element))

    plt.scatter(x, y)
    plt.scatter(x, y_predicted, marker='*', c='red')

    plt.show()



def main():
    # todo4()
    # todo5()
    # todo6()
    # todo7()
    todo8()

if __name__ == '__main__':
    main()

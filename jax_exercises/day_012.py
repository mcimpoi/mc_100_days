# Perceptron implementation in JAX;
# from Machine Learning with Pytorch and Scikit-Learn; (chapter 2)

# https://github.com/rasbt/machine-learning-book/blob/main/ch02/ch02.ipynb
# adapted from numpy to jax; following the book implementation

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for WSL
import matplotlib.pyplot as plt
import pandas as pd


class Perceptron:
    def __init__(self, eta=0.01, n_iter=10, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_ = None
        self.b_ = None
        self.mean_ = 0.0
        self.std_ = 0.01
        self.errors_ = []

    def fit(self, X, y):
        key = jax.random.PRNGKey(self.random_state)
        self.w_ = jax.random.normal(key, shape=(1, X.shape[1])) * self.std_ + self.mean_
        self.b_ = 0.0
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int((update != 0.0).astype(int)[0])
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return jnp.dot(X, self.w_.T) + self.b_

    def predict(self, X):
        return jnp.where(self.net_input(X) >= 0.0, 1, -1)


def plot_data(X):
    plt.figure(figsize=(8, 6))
    plt.scatter(
        X[:50, 0], X[:50, 1], c="red", marker="o", label="setosa"
    )
    plt.scatter(
        X[50:100, 0],
        X[50:100, 1],
        c="blue",
        marker="x",
        edgecolor="black",
        label="versicolor",
    )
    plt.xlabel("sepal length [cm]")
    plt.ylabel("petal length [cm]")
    plt.legend(loc="upper left")
    plt.savefig('iris_plot.png')  # Save the plot to a file
    plt.close()  # Close the figure to free memory


def main():
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
        encoding="utf-8",
    )

    y = df.iloc[0:100, 4].values
    y = jnp.where(y == "Iris-setosa", -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    plot_data(X)

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)

    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Number of misclassifications")
    plt.savefig("perceptron_errors.png")
    plt.close()

if __name__ == "__main__":
    main()

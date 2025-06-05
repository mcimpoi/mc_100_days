import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for WSL
import matplotlib.pyplot as plt 
import pandas as pd

class AdalineGD:
    
    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_ = None
        self.b_ = None
        self.losses_ = []

    def fit(self, X: jnp.ndarray, y: jnp.ndarray):
        key = jax.random.PRNGKey(self.random_state)
        self.w_ = jax.random.normal(key, (X.shape[1], 1)) * 0.01
        self.b_ = jnp.zeros(1)
        self.losses_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_ += self.eta * 2.0 * jnp.dot(X.T, errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * jnp.mean(errors)
            loss = jnp.mean(errors ** 2) 
            self.losses_.append(loss)
        return self

    def net_input(self, X: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(X, self.w_) + self.b_
    
    def activation(self, X: jnp.ndarray) -> jnp.ndarray:
        return X
    
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    
if __name__ == "__main__":
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
        encoding="utf-8",
    )

    y = df.iloc[0:100, 4].values
    y = jnp.where(y == "Iris-setosa", -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    ada = AdalineGD(n_iter=15, eta=0.1).fit(X, y)
    ax[0].plot(range(1, len(ada.losses_) + 1), jnp.log(jnp.array(ada.losses_)), marker="o")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Log Mean squared error")
    ax[0].set_title("Adaline - Learning rate 0.1")

    ada = AdalineGD(n_iter=15, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada.losses_) + 1), ada.losses_, marker="o")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Mean squared error")
    ax[1].set_title("Adaline - Learning rate 0.0001")

    plt.tight_layout()
    plt.savefig("adaline_gd.png")
    plt.close()
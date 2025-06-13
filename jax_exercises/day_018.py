import jax
import jax.numpy as jnp
from tqdm import tqdm

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# TODO: this is slow. Understand why and @jax.jit

class AdalineSGD:
    def __init__(self, learning_rate=0.01, n_iterations=10, random_state=None, shuffle=True):
        
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.shuffle = shuffle
        self.random_key = jax.random.PRNGKey(random_state or 0)
        self.w_ = None
        self.b_ = None
        self.feature_scaling = 0.01

    def fit(self, X, y):
        self.losses_ = []
        self._initialize_weights(X.shape[1])
        for _ in tqdm(range(self.n_iterations)):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in tqdm(zip(X, y)):
                losses.append(self._update_weights(xi, target))
                avg_loss = jnp.mean(jnp.array(losses))
                self.losses_.append(avg_loss)
        return self
    
    def partial_fit(self, X, y):
        if self.w_ is None or self.b_ is None:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self
    
    def _initialize_weights(self, n_features):
        w_key, b_key = jax.random.split(self.random_key)
        self.w_ = jax.random.normal(w_key, (n_features,)) * self.feature_scaling
        self.b_ = jax.random.normal(b_key, (1,))

    def _shuffle(self, X, y):
        perm_key, _ = jax.random.split(self.random_key)
        perm = jax.random.permutation(perm_key, jnp.arange(X.shape[0]))
        return X[perm], y[perm]
    
    def _update_weights(self, xi, target):
        
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_ += 2.0 * self.learning_rate * error * xi
        self.b_ += 2.0 * self.learning_rate * error
        return jnp.square(error)
    
    def net_input(self, X):
        return jnp.dot(X, self.w_) + self.b_

    def activation(self, X):
        return X
    
    def predict(self, X):
        return jnp.where(self.net_input(X) >= 0.5, 1, 0)
    
if __name__ == "__main__":

    # Create a synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to JAX arrays
    X_train_jax = jnp.array(X_train)
    y_train_jax = jnp.array(y_train)
    X_test_jax = jnp.array(X_test)
    y_test_jax = jnp.array(y_test)

    # Initialize and train the model
    model = AdalineSGD(learning_rate=0.01, n_iterations=10, random_state=42)
    model.fit(X_train_jax, y_train_jax)

    # Make predictions
    predictions = model.predict(X_test_jax)
    
    # Calculate accuracy
    accuracy = jnp.mean(predictions == y_test_jax)
    print(f"Accuracy: {accuracy:.2f}")
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state


class SimpleCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x
    


class TrainState(train_state.TrainState):
    pass


if __name__ == "__main__":
    key = jax.random.key(0)
    input_shape = (1, 28, 28, 1)

    model = SimpleCNN()
    params = model.init(key, jnp.ones(input_shape))["params"]

    print(jax.tree.map(lambda x: x.shape, params))

    dummy_img = jnp.ones(input_shape)
    logits = model.apply({'params': params}, dummy_img)
    print(logits)
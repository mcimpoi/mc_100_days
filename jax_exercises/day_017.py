import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import tensorflow_datasets as tfds
import tensorflow as tf

from tqdm import tqdm

# Ensure TensorFlow does not allocate GPU memory.
tf.config.experimental.set_visible_devices([], "GPU")

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
    

def train_step(state, batch, num_classes):
    
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["image"])
        one_hot_labels = jax.nn.one_hot(batch["label"], num_classes=num_classes)

        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels))
        accuracy = jnp.mean(jnp.argmax(logits, -1) == batch["label"])

        return loss, accuracy
    
    grad_fn = jax.value_and_grad(loss_fn,  has_aux=True)
    (loss, accuracy), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)
    metrics = {"loss": loss, "accuracy": accuracy}
    return state, metrics

def mnist_get_train_batches(data_dir, variant="mnist", batch_size=128):
    ds = tfds.load(name=variant, split="train", as_supervised=True, data_dir=data_dir)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(1)
    return tfds.as_numpy(ds)


if __name__ == "__main__":
    key = jax.random.key(0)
    input_shape = (1, 28, 28, 1)

    model = SimpleCNN()
    params = model.init(key, jnp.ones(input_shape))["params"]

    print(jax.tree.map(lambda x: x.shape, params))

    dummy_img = jnp.ones(input_shape)
    logits = model.apply({'params': params}, dummy_img)
    print(logits)

    num_epochs = 10
    data_dir = "/data/minst"
    batch_size = 64

    num_classes = 10
    tx = optax.adam(learning_rate=0.001)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    for epoch in tqdm(range(1, num_epochs + 1)):
        epoch_loss = []
        epoch_accuracy = []

        for data_batch in mnist_get_train_batches(data_dir=data_dir, variant="fashion_mnist", batch_size=batch_size):
            batch = {"image": data_batch[0], "label": data_batch[1]}
            state, metrics = train_step(state, batch, num_classes)
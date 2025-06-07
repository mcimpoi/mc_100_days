# https://docs.jax.dev/en/latest/notebooks/neural_network_with_tfds_data.html

import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds

DEFAULT_SEED = 42


def random_layer_params(m, n, key, scale=1e-2) -> tuple[jnp.ndarray, jnp.ndarray]:
    w_key, b_key = jax.random.split(key)
    return scale * jax.random.normal(w_key, (n, m)), scale * jax.random.normal(
        b_key, (n,)
    )


def init_network_params(layer_sizes, key):
    keys = jax.random.split(key, len(layer_sizes))
    return [
        random_layer_params(m, n, k)
        for m, n, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)
    ]


def relu(x):
    return jnp.maximum(0, x)


def predict(params, image):
    # per example predictions
    activations = image  # shape: (m,)
    for w, b in params[:-1]:
        # w shape: (n, m), activations shape: (m,)
        # jnp.dot(w, activations) shape: (n,)
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return jax.nn.log_softmax(logits)


def one_hot(x, k, dtype=jnp.float32):
    return jnp.eye(k, dtype=dtype)[x]


def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(target_class == predicted_class)


def loss(params, images, targets):
    preds = batched_predict(params, images)
    return -jnp.mean(targets * preds)


def update(params, x, y, step_size):
    grads = jax.grad(loss)(params, x, y)
    return [
        (w - step_size * dw, b - step_size * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]


def mnist_get_train_batches(data_dir, batch_size=128):
    ds = tfds.load("mnist", split="train", as_supervised=True, data_dir=data_dir)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(1)
    return tfds.as_numpy(ds)


if __name__ == "__main__":
    layer_sizes = [784, 512, 512, 10]
    step_size = 0.01
    num_epochs = 10
    batch_size = 128
    n_targets = 10

    key = jax.random.key(DEFAULT_SEED)
    params = init_network_params(layer_sizes, jax.random.key(DEFAULT_SEED))

    # Print device information
    print("Available devices:", jax.devices())
    print("Default backend:", jax.default_backend())
    print("Current device:", jax.devices()[0])

    # works on single example
    random_flattened_image = jax.random.normal(key, (784,))
    preds = predict(params, random_flattened_image)
    print(preds)

    random_flattened_batch = jax.random.normal(key, (batch_size, 784))
    try:
        preds = predict(params, random_flattened_batch)
    except TypeError as e:
        print(f"Invalid shape: {e}")

    batched_predict = jax.vmap(predict, in_axes=(None, 0))
    preds = batched_predict(params, random_flattened_batch)
    print(f"Works on batch: {preds.shape}")

    data_dir = "/tmp/datasets/"

    mnist_data, info = tfds.load(
        name="mnist", batch_size=-1, data_dir=data_dir, with_info=True
    )
    print(info)

    mnist_data = tfds.as_numpy(mnist_data)
    train_data, test_data = mnist_data["train"], mnist_data["test"]
    num_labels = info.features["label"].num_classes
    h, w, c = info.features["image"].shape
    num_pixels = h * w * c

    # full training set
    train_images = train_data["image"]
    train_labels = train_data["label"]
    train_images = jnp.reshape(train_images, (train_images.shape[0], num_pixels))
    train_labels = one_hot(train_labels, num_labels)

    # full test set
    test_images = test_data["image"]
    test_labels = test_data["label"]
    test_images = jnp.reshape(test_images, (test_images.shape[0], num_pixels))
    test_labels = one_hot(test_labels, num_labels)

    print(train_images.shape, train_labels.shape)
    print(test_images.shape, test_labels.shape)

    # train
    for epoch in range(num_epochs):
        for x, y in mnist_get_train_batches(data_dir):
            x = jnp.reshape(x, (x.shape[0], num_pixels))
            y = one_hot(y, num_labels)
            params = update(params, x, y, step_size)

        train_acc = accuracy(params, train_images, train_labels)
        test_acc = accuracy(params, test_images, test_labels)
        print(
            f"Epoch {epoch}: Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}"
        )

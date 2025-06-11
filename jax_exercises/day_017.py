import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import tensorflow_datasets as tfds
import tensorflow as tf
import time 

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

# Create a JIT-compiled version of train_step
# Mark 'num_classes' as a static argument for JIT compilation
train_step_jit = jax.jit(train_step, static_argnames=('num_classes',))

def mnist_get_train_batches(data_dir, variant="mnist", batch_size=128):
    ds = tfds.load(name=variant, split="train", as_supervised=True, data_dir=data_dir)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(1)
    return tfds.as_numpy(ds)

def profile_training(data_dir, dataset_variant, batch_size, initial_state, num_classes):
    print("\\n--- Starting Profiling ---")
    num_profiling_steps = 50  # Number of steps to run for profiling
    
    # Get a single batch for profiling
    profiling_data_iterator = mnist_get_train_batches(data_dir=data_dir, variant=dataset_variant, batch_size=batch_size)
    profiling_batch_tf = next(iter(profiling_data_iterator))
    profiling_batch = {"image": profiling_batch_tf[0], "label": profiling_batch_tf[1]}

    # Profile non-JIT version
    state_nojit = initial_state # Use a fresh state
    start_time_nojit = time.time()
    for _ in range(num_profiling_steps):
        state_nojit, metrics_nojit = train_step(state_nojit, profiling_batch, num_classes)
        metrics_nojit["loss"].block_until_ready() # Ensure computation finishes
    end_time_nojit = time.time()
    duration_nojit = end_time_nojit - start_time_nojit
    print(f"Time for {num_profiling_steps} steps (no JIT): {duration_nojit:.4f} seconds")

    # Profile JIT version
    state_jit = initial_state # Use a fresh state
    # Warm-up call for JIT
    print("Warming up JIT compiled function...")
    state_jit, metrics_jit_warmup = train_step_jit(state_jit, profiling_batch, num_classes)
    metrics_jit_warmup["loss"].block_until_ready() # Ensure compilation and first run finish
    
    start_time_jit = time.time()
    for _ in range(num_profiling_steps):
        state_jit, metrics_jit = train_step_jit(state_jit, profiling_batch, num_classes)
        metrics_jit["loss"].block_until_ready() # Ensure computation finishes
    end_time_jit = time.time()
    duration_jit = end_time_jit - start_time_jit
    print(f"Time for {num_profiling_steps} steps (with JIT): {duration_jit:.4f} seconds")
    
    if duration_nojit > 0 and duration_jit > 0:
        speedup = duration_nojit / duration_jit
        print(f"JIT speedup: {speedup:.2f}x")
    print("--- Profiling Finished ---\\n")

if __name__ == "__main__":
    key = jax.random.key(0)
    input_shape = (1, 28, 28, 1)

    model = SimpleCNN()
    params = model.init(key, jnp.ones(input_shape))["params"]

    print(f"model parameter shapes {jax.tree.map(lambda x: x.shape, params)}")


    num_epochs = 10
    data_dir = "/data/minst"
    batch_size = 64
    dataset_variant = "fashion_mnist" 

    num_classes = 10
    learning_rate: float = 0.001
    tx = optax.adam(learning_rate=learning_rate)
    
    # Initial state for profiling and training
    initial_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    profile_training(data_dir, dataset_variant, batch_size, initial_state, num_classes)

    print(f"Starting training for {num_epochs} epochs using JIT compiled train_step...")
    current_state = initial_state # Reset state for the actual training
    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs"):
        epoch_loss = []
        epoch_accuracy = []

        train_batches = mnist_get_train_batches(data_dir=data_dir, variant=dataset_variant, batch_size=batch_size)
        for data_batch in tqdm(train_batches, desc=f"Epoch {epoch}/{num_epochs}", unit="batch", leave=False):
            batch = {"image": data_batch[0], "label": data_batch[1]}
            current_state, metrics = train_step_jit(current_state, batch, num_classes) 
            epoch_loss.append(metrics["loss"].item()) # .item() to get Python scalar
            epoch_accuracy.append(metrics["accuracy"].item())
        
        avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0
        avg_accuracy = sum(epoch_accuracy) / len(epoch_accuracy) if epoch_accuracy else 0
        print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")


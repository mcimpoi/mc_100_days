# https://docs.jax.dev/en/latest/sharded-computation.html

# run on CPU with
#  JAX_PLATFORMS="cpu" python jax_exercises/day_016.py
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P


@jax.jit
def f_elementwise(x):
    return 2 * jnp.sin(x) + 1


@jax.jit
def f_contract(x):
    return x.sum(axis=0)


if __name__ == "__main__":
    jax.config.update("jax_num_cpu_devices", 8)
    jax.default_device("cpu")
    print(f"{jax.devices()=}")

    arr = jnp.arange(32.0).reshape(4, 8)
    print(f"{arr.devices()=}")
    print(f"{arr.sharding=}")
    jax.debug.visualize_array_sharding(arr)

    mesh = jax.make_mesh((2, 4), ("x", "y"))
    print(f"{mesh=}")

    sharding = jax.sharding.NamedSharding(mesh, P("x", "y"))
    print(f"{sharding=}")

    arr_sharded = jax.device_put(arr, sharding)
    print(f"{arr_sharded=}")

    jax.debug.visualize_array_sharding(arr_sharded)

    result = f_elementwise(arr_sharded)
    print("shardings match:", result.sharding == arr_sharded.sharding)

    result = f_contract(arr_sharded)
    jax.debug.visualize_array_sharding(result)
    print(f"{result=}")
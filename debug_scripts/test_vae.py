import jax
import jax.numpy as jnp
from PIL import Image
import numpy as np
import flax
from flax.training import checkpoints
import functools

from bageljax.autoencoder import build_autoencoder

flax.config.update('flax_use_orbax_checkpointing', False)

# Load the source image
src_img = Image.open("colors.png")
src_img_np = np.array(src_img)
assert src_img_np.shape == (1024, 1024, 3)
assert src_img_np.dtype == np.uint8
src_img_np = 2 * (src_img_np.astype(np.float32) / 255.0) - 1
src_img_np = src_img_np[None, ...] # add a batch dimension
src_img_np = jnp.array(src_img_np, dtype=jnp.float32)

# Load the autoencoder
def make_jitted_encode(ae):
    """Return a jit-compiled encode(X) -> z."""
    @functools.partial(jax.jit,
                       static_argnames=("method",))
    def _encode(variables, x, key, *, method):
        return ae.apply(variables, x,
                        rngs={"gaussian": key},
                        method=method)
    # bind the method so callers don’t pass it
    return functools.partial(_encode, method=ae.encode)

def make_jitted_decode(ae):
    """Return a jit-compiled decode(z) -> X_rec."""
    @functools.partial(jax.jit,
                       static_argnames=("method",))
    def _decode(variables, z, *, method):
        return ae.apply(variables, z, method=method)
    return functools.partial(_decode, method=ae.decode)

ae = build_autoencoder(sample_latent=True)
rng = jax.random.PRNGKey(0)
rng, key = jax.random.split(rng)
ae_variables = ae.init(key, jnp.zeros((1, 1024, 1024, 3)))
print("Autoencoder initialized.")

# Load weights for the autoencoder
ae_checkpoint_path = "/home/pranav/bageljax/new_ae_ckpt"
ae_variables = checkpoints.restore_checkpoint(ae_checkpoint_path, target=ae_variables)
print("Autoencoder weights loaded.")

ae_encode = make_jitted_encode(ae)
ae_decode = make_jitted_decode(ae)

# Encode and decode test
rng, key = jax.random.split(rng)
img_encoded = ae_encode(ae_variables, src_img_np, key)
assert img_encoded.shape == (1, 128, 128, 16) and img_encoded.dtype == jnp.float32

img_decoded = ae_decode(ae_variables, img_encoded)
img_decoded = np.array(img_decoded)[0]
img_decoded = np.clip((img_decoded + 1) * 127.5, 0, 255).astype(np.uint8)
recon = Image.fromarray(img_decoded)

recon.save("recon.png")
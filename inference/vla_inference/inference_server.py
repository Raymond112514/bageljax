import jax
jax.distributed.initialize()

from functools import partial
import jax.numpy as jnp
import numpy as np
import flax
from flax.core import FrozenDict
import orbax.checkpoint as ocp
from etils import epath
from typing import Dict
from PIL import Image
import asyncio
import dataclasses
import logging
import traceback
import websockets.asyncio.server
import websockets.frames

from bageljax.common.common import TrainState, ModuleDict, nonpytree_field
from bageljax.utils.jax_utils import create_sharding, add_batch_sharding_constraint, enforce_sharding_constraints
from bageljax.model.vocabulary import TokenEmbedder, LogitsHead
from bageljax.model.vision_encoder import VisionEncoder
from bageljax.model.mixture_of_transformers import MixtureOfTransformers
from bageljax.model.tokenizer import Qwen2Tokenizer, add_special_tokens
from bageljax.utils import msgpack_numpy

INFERENCE_CONFIG = {
    "seed": 0,
    "checkpoint_load_dir": "gs://path-to-your-value-function-checkpoint",
    "tokenizer_load_path": "/path/to/your/tokenizer",
    "max_prompt_length": 226,
}

# Create the language tokenizer
tokenizer_load_path = INFERENCE_CONFIG["tokenizer_load_path"]
tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_load_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

# Function to convert text instruction strings into padded tokens, and adds to the batch 
# these tokens as well as the token masks and text rope IDs
def tokenize_and_pad(batch):
    # What this function is going to do is rewrite the language instruction into:
    # "How many timesteps away is the robot from successfully completing the following language instruction:\n\n<language_instruction>\n\nDistance:"
    # and then tokenize. It will also left-pad the text to the max length, and assign RoPE IDs.

    PAD_TOKEN_ID = 0 # it doesn't really matter what this is
    MAX_PROMPT_LENGTH = FLAGS.config.dataset_kwargs["max_prompt_length"]

    B = batch["image"].shape[0]
    batch_tokenized_language = []
    batch_masks = []
    batch_text_rope_ids = []
    for i in range(B):
        prompt = "How many timesteps away is the robot from successfully completing the following language instruction:\n\n"
        prompt += batch["language_instruction"][i].decode("utf-8").strip()
        prompt += "\n\nDistance:"
        prompt_text_ids = tokenizer.encode(prompt)
            
        # Add bos and eos tokens
        prompt_text_ids = [new_token_ids['bos_token_id']] + prompt_text_ids + [new_token_ids['eos_token_id']]

        # Also add the bos token at the end to start value prediction
        prompt_text_ids = prompt_text_ids + [new_token_ids['bos_token_id']]

        non_pad_tokens_in_prompt = len(prompt_text_ids)
        assert non_pad_tokens_in_prompt < MAX_PROMPT_LENGTH

        # Pad on the left
        num_pad_tokens = MAX_PROMPT_LENGTH - non_pad_tokens_in_prompt
        prompt_text_ids = [PAD_TOKEN_ID] * num_pad_tokens + prompt_text_ids
        prompt_text_ids = np.array(prompt_text_ids, dtype=np.int32)
        masks = np.concatenate([np.zeros((num_pad_tokens,), dtype=bool), np.ones((non_pad_tokens_in_prompt,), dtype=bool)]) # false means padding, true means valid text token
        text_rope_ids = np.concatenate([
            np.zeros((num_pad_tokens,), dtype=np.int32), # the rope IDs for pad tokens doesn't matter
            np.arange(non_pad_tokens_in_prompt, dtype=np.int32) + 1, # start from 1, since the image comes before
        ])

        batch_tokenized_language.append(prompt_text_ids)
        batch_masks.append(masks)
        batch_text_rope_ids.append(text_rope_ids)

    batch_tokenized_language = np.stack(batch_tokenized_language)
    batch_masks = np.stack(batch_masks)
    batch_text_rope_ids = np.stack(batch_text_rope_ids)

    del batch["language_instruction"]
    batch["text_tokens"] = batch_tokenized_language
    batch["text_token_masks"] = batch_masks
    batch["text_rope_ids"] = batch_text_rope_ids
    return batch

rng = jax.random.PRNGKey(INFERENCE_CONFIG["seed"])
enforce_sharding_constraints(False)

# Initialize the main model
print("Initializing Bagel Value Function model...")

networks = {
    "token_embedder": TokenEmbedder(),
    "vision_encoder": VisionEncoder(),
    "mixture_of_transformers": MixtureOfTransformers(),
    "logits_head": LogitsHead(),
}

model_def = ModuleDict(networks)

def init_fn(rng):
    rng, init_rng = jax.random.split(rng)

    # For init, let's pick reasonable values of some of the input parameters
    B = 1 # batch size
    H, W = 672, 672 # image height and width, 672 is divisible by 14 and 16 ---> no longer needs to be divisible by 16
    llm_hidden_dim = 3584 # LLM hidden dimension
    L = 256 # needs to be a multiple of 128 for flash attention

    params = model_def.init(
        {"params": init_rng},
        token_embedder=[
            jnp.zeros((B, L), dtype=jnp.int32),
        ],
        vision_encoder=[
            jnp.zeros((B, H, W, 3), dtype=jnp.bfloat16),
        ],
        mixture_of_transformers=[
            jnp.zeros((B, L, llm_hidden_dim), dtype=jnp.bfloat16),
            jnp.zeros((B, L), dtype=jnp.int32),
            jnp.zeros((B, 1, L, L), dtype=jnp.bfloat16),
        ],
        logits_head=[
            jnp.zeros((B, L, llm_hidden_dim), dtype=jnp.bfloat16),
        ],
    )["params"]
    rng, create_rng = jax.random.split(rng)
    train_state = TrainState.create(
        apply_fn=model_def.apply,
        params=params,
        txs=None,
        target_params=None,
        rng=create_rng,
    )

    return train_state

rng, key = jax.random.split(rng)
train_state_shape = jax.eval_shape(init_fn, key)

# Create sharding and train_state
data_sharding, train_state_sharding, no_shard, shard_data, global_to_local = create_sharding("fsdp", train_state_shape)
rng, key = jax.random.split(rng)
train_state = jax.jit(init_fn, out_shardings=train_state_sharding)(key)

# Create checkpointer object (used for all subsequent checkpoint loading)
checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())

# Load for trained Bagel value function checkpoint
loaded_value_params = checkpointer.restore(INFERENCE_CONFIG["checkpoint_load_dir"], train_state.params)
train_state = train_state.replace(params=loaded_value_params)
print("Loaded value function checkpoint")
print("Total model parameters: ", sum([np.prod(v.shape) for v in jax.tree_util.tree_leaves(train_state.params)]))

config = flax.core.FrozenDict(dict())


# Define the inference function
@partial(
    jax.jit,
    in_shardings=(train_state_sharding, data_sharding),
    out_shardings=data_sharding,
)
def infer(train_state, batch):
    # Prepare the vit tokens
    image = jnp.astype(batch["image"], jnp.float32) / 127.5 - 1
    image = jnp.astype(image, jnp.bfloat16)
    image = add_batch_sharding_constraint(image, where="image before vit")

    # Pass through the ViT
    pre_llm_vit_tokens = train_state.apply_fn(
        {"params": train_state.params},
        img=image,
        name="vision_encoder",
    )
    pre_llm_vit_tokens = add_batch_sharding_constraint(pre_llm_vit_tokens, where="image after vit")

    # Let's embed the start and end image tokens
    image_special_token_ids = jnp.array([new_token_ids["start_of_image"], new_token_ids["end_of_image"]], dtype=jnp.int32)
    image_special_token_embeds = train_state.apply_fn({"params": train_state.params},token_ids=image_special_token_ids[None, :], name="token_embedder")
    image_special_token_embeds = jnp.tile(image_special_token_embeds, (pre_llm_vit_tokens.shape[0], 1, 1))
    image_special_token_embeds = add_batch_sharding_constraint(image_special_token_embeds, where="image_special_token_embeds")

    # concat the special image tokens
    pre_llm_vit_tokens = jnp.concatenate([image_special_token_embeds[:, 0:1], pre_llm_vit_tokens, image_special_token_embeds[:, 1:2]], axis=1)
    pre_llm_vit_tokens = add_batch_sharding_constraint(pre_llm_vit_tokens, where="pre_llm_vit_tokens")

    # embed the text tokens
    text_embeds = train_state.apply_fn(
        {"params": train_state.params},
        token_ids=batch["text_tokens"],
        name="token_embedder",
    )
    text_embeds = add_batch_sharding_constraint(text_embeds, where="text_embeds")

    # Concat everything along the sequence dimension
    full_seq = jnp.concatenate([pre_llm_vit_tokens, text_embeds], axis=1)
    full_seq = add_batch_sharding_constraint(full_seq, where="full_seq")

    # Prepare the full seq rope ids
    image_rope_ids = jnp.zeros((pre_llm_vit_tokens.shape[1],), dtype=jnp.int32)[None, :]
    image_rope_ids = jnp.tile(image_rope_ids, (full_seq.shape[0], 1))

    full_seq_rope_ids = jnp.concatenate([image_rope_ids, batch["text_rope_ids"]], axis=1)
    full_seq_rope_ids = add_batch_sharding_constraint(full_seq_rope_ids, where="full_seq_rope_ids")

    # Now prepare attention masks
    L = full_seq.shape[1]
    # Start with a causal mask
    causal = jnp.tril(jnp.ones((L, L), dtype=bool))
    # Enable full self-attention for vit tokens
    arr = jnp.concatenate([jnp.ones((pre_llm_vit_tokens.shape[1],), dtype=bool), jnp.zeros((L - pre_llm_vit_tokens.shape[1]), dtype=bool)])
    self_attention_mask = jnp.matmul(arr[:, None], arr[None, :])
    block_causal = causal | self_attention_mask
    # Now tile to include batch dimension
    block_causal = jnp.tile(block_causal[None, ...], (full_seq.shape[0], 1, 1))
    block_causal = add_batch_sharding_constraint(block_causal, where="block_causal")
    # padding sees nothing, and is seen by nothing
    padding = jnp.concatenate([jnp.zeros((full_seq.shape[0], pre_llm_vit_tokens.shape[1]), dtype=bool), jnp.logical_not(batch["text_token_masks"])], axis=1)
    padding = add_batch_sharding_constraint(padding, where="padding")
    allowed_attention = jnp.where(padding[:, :, None] | padding[:, None, :], False, block_causal)
    attn_bias = jnp.where(allowed_attention, 0.0, -1e30)[:, None, :, :]   # (B,1,L,L)
    attn_bias = attn_bias.astype(jnp.bfloat16) # mixed precision is annoying, lol
    attn_bias = add_batch_sharding_constraint(attn_bias, where="attn_bias")

    # Now feed through the LLM
    post_llm_seq = train_state.apply_fn(
        {"params": train_state.params},
        x=full_seq,
        rope_pos_ids=full_seq_rope_ids,
        attn_bias=attn_bias,
        name="mixture_of_transformers",
    )
    post_llm_seq = add_batch_sharding_constraint(post_llm_seq, where="post_llm_seq")

    # Extract just the last token
    post_llm_value_token = post_llm_seq[:, -1:]

    # Feed into LLM head
    value_logits = train_state.apply_fn(
        {"params": train_state.params},
        hidden_states=post_llm_value_token,
        name="logits_head",
    )
    # this should have shape (B, 1, 512)
    assert value_logits.shape == (post_llm_seq.shape[0], 1, 512)
    assert value_logits.dtype == jnp.bfloat16
    value_logits = add_batch_sharding_constraint(value_logits, where="value_logits")
    # get rid of singleton dimension
    value_logits = value_logits[:, 0]
    
    return value_logits

enforce_sharding_constraints(True)

class ValueFunction:
    def __init__(self, rng):
        self.rng = rng
    
    def infer(self, obs: Dict) -> Dict:
        batch = self.preprocess_obs(obs)
        batch = tokenize_and_pad(batch)
        batch = shard_data(batch)

        value_logits = infer(train_state, batch)
        value_logits = np.array(jax.device_get(value_logits))
        assert value_logits.shape[0] == jax.device_count()

        logits = value_logits[0, 0]

        return {"logits": logits}
    
    def preprocess_obs(self, obs: Dict) -> Dict:
        shoulder = obs["shoulder_image"]
        wrist = obs["wrist_image"]
        instruction = obs["language_instruction"]
        
        # Concatenate the shoulder and wrist images, copied from dataset.py
        shoulder_and_wrist = tf.concat([shoulder, wrist], axis=0)
        shoulder_and_wrist = tf.ensure_shape(shoulder_and_wrist, [576, 512, 3])
        shoulder_and_wrist = tf.cast(tf.round(tf.image.resize(shoulder_and_wrist, (672, 560), method="bicubic")), tf.uint8)
        return {"image": shoulder_and_wrist, "language_instruction": instruction}

    def reset(self, reset_info: Dict) -> None:
        pass

@dataclasses.dataclass
class ValueFunctionServerConfig:
    image_resolution: tuple[int, int] | None = (672, 560)
    needs_wrist_camera: bool = True
    n_external_cameras: int = 1
    needs_stereo_camera: bool = False
    needs_session_id: bool = False
    action_space: str = "joint_position"

class WebsocketValueFunctionServer:
    def __init__(
        self,
        value_function: ValueFunction,
        server_config: ValueFunctionServerConfig,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        self._value_function = value_function
        self._server_config = server_config
        self._host = host
        self._port = port
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(dataclasses.asdict(self._server_config)))

        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                value = self._value_function.infer(obs)
                to_return = packer.pack(value["logits"])
                await websocket.send(to_return)
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

rng, value_function_rng = jax.random.split(rng)
value_function = ValueFunction(rng=value_function_rng)
server_config = ValueFunctionServerConfig(
    image_resolution=(672, 560),
    needs_wrist_camera=True,
    n_external_cameras=1,
)
server = WebsocketValueFunctionServer(value_function, server_config)
server.serve_forever()

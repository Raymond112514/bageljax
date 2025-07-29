"""
Note the file tokenizer.py under checkpoints is used inexplicably. Make sure to figure this out when porting over to JAX.

I think I figured it out. See https://chatgpt.com/c/687c2c82-12c8-800f-9e78-be1a7fb88bfa
"""

from bageljax.tokenizer import Qwen2Tokenizer, add_special_tokens

tokenizer_load_path = "pretrained_weights/tokenizer"
tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_load_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

print("BOS token ID:", new_token_ids['bos_token_id'])
print("EOS token ID:", new_token_ids['eos_token_id'])
print("Start of image token ID:", new_token_ids['start_of_image'])
print("End of image token ID:", new_token_ids['end_of_image'])
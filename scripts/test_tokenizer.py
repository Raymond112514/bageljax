from bageljax.tokenizer import Qwen2Tokenizer, add_special_tokens

tokenizer_load_path = "pretrained_weights/tokenizer"
tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_load_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

print("BOS token ID:", new_token_ids['bos_token_id'])
print("EOS token ID:", new_token_ids['eos_token_id'])
print("Start of image token ID:", new_token_ids['start_of_image'])
print("End of image token ID:", new_token_ids['end_of_image'])
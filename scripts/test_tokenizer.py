from bageljax.tokenizer import Qwen2Tokenizer, add_special_tokens

tokenizer_load_path = "pretrained_weights/tokenizer"
tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_load_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
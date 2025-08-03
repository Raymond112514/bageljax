from bageljax.tokenizer import Qwen2Tokenizer, add_special_tokens

tokenizer_load_path = "/home/pranav/bageljax/pretrained_weights/tokenizer"
tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_load_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

print("BOS token ID:", new_token_ids['bos_token_id'])
print("EOS token ID:", new_token_ids['eos_token_id'])
print("Start of image token ID:", new_token_ids['start_of_image'])
print("End of image token ID:", new_token_ids['end_of_image'])

sample_text = "so long and thanks for all the fish"
text_ids = tokenizer.encode(sample_text)

print("text ids:", text_ids)
# text ids: [704, 1293, 323, 9339, 369, 678, 279, 7640]
# this means the bos and eos tokens have not been added (which is what we expect)

print(type(text_ids))
# <class 'list'>, nice
from bageljax.model.tokenizer import Qwen2Tokenizer, add_special_tokens

tokenizer_load_path = "/home/pranav/bageljax/pretrained_weights/tokenizer"
tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_load_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

sample_text = "What actions should the robot take to complete the following language instruction:\n\n<language_instruction>\n\nActions:"
text_ids = tokenizer.encode(sample_text)

print("text ids:", text_ids)
# text ids: [704, 1293, 323, 9339, 369, 678, 279, 7640]
# this means the bos and eos tokens have not been added (which is what we expect)

print(len(text_ids))
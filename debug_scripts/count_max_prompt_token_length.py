from bageljax.tokenizer import Qwen2Tokenizer, add_special_tokens
from tqdm import tqdm
import json
import numpy as np

tokenizer_load_path = "/home/pranav/bageljax/pretrained_weights/tokenizer"
tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_load_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

with open("droid_language_annotations.json") as f:
    all_language_annotations = json.load(f)
all_strings = []
for key in all_language_annotations.keys():
    l1 = all_language_annotations[key]["language_instruction1"]
    l2 = all_language_annotations[key]["language_instruction2"]
    l3 = all_language_annotations[key]["language_instruction3"]
    for l in [l1, l2, l3]:
        if l != "":
            all_strings.append(l)

lengths = []
max_length, max_length_s = -1, ""
min_length, min_length_s = 100, ""
for s in tqdm(all_strings):
    text_ids = tokenizer.encode(s)
    lengths.append(len(text_ids))
    if len(text_ids) > max_length:
        max_length = len(text_ids)
        max_length_s = s
    elif len(text_ids) < min_length:
        min_length = len(text_ids)
        min_length_s = s

lengths = np.array(lengths, dtype=np.int32)
lengths = sorted(lengths, reverse=True)
print(lengths[:20])
print("Max:", np.max(lengths))
print("Min:", np.min(lengths))
print("Mean:", np.mean(lengths))
print("Max length string:", max_length_s)
print("Min length string:", min_length_s)
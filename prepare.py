from datasets import load_dataset
from transformers import GPT2TokenizerFast
import numpy as np
import pickle
import os

# initializing file path

save_dir = "data/mini_openwebtext_3"
os.makedirs(save_dir, exist_ok=True)

train_bin_path = os.path.join(save_dir, "train.bin")
val_bin_path   = os.path.join(save_dir, "val.bin")
meta_path      = os.path.join(save_dir, "meta.pkl")

# download the first 3% of the dataset
# python dict with a 'text' field
print('Loading dataset')
dataset = load_dataset("roneneldan/TinyStories", split="train[:5%]")

# load gpt-2 tokenizer (BPE) which produces token IDs in [0, 50256]
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

# tokenizing dataset
print('Tokenizing text')
all_ids = []

# loop and convert to sequence of token
for x in dataset['text']:
    # encode(x) returns list of int toekn IDs
    all_ids.extend(tokenizer.encode(x))

# convert to array
ids = np.array(all_ids, dtype=np.uint16)

n = int(0.9 * len(ids)) # 90% train

# split
train_data = ids[:n]
val_data = ids[n:]

# save train and val
train_data.tofile(train_bin_path)
val_data.tofile(val_bin_path)

with open(meta_path, "wb") as f:
    pickle.dump(
        {
            "vocab_size": tokenizer.vocab_size,
            "eos_token_id": tokenizer.eos_token_id,
        },
        f,
    )

print('DONE, saved in:', save_dir)



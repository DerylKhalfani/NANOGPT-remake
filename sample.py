import torch
import pickle

from transformers import GPT2TokenizerFast

from model import GPT, GPTConfig  # your classes

device = "cuda" if torch.cuda.is_available() else "cpu"

# direction path
data_dir = "data/mini_openwebtext_3"   # same as prepare.py
ckpt_path = "model_final.pt"

# load meta which includes vocab_size and eos token id
with open(f"{data_dir}/meta.pkl", "rb") as f:
    meta = pickle.load(f)

vocab_size   = meta["vocab_size"]
eos_token_id = meta["eos_token_id"]

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Initialize model IMPORTANT TO MATCH TRAINING.py
config = GPTConfig()

model = GPT(config).to(device)

# Load checkpoint (covers both "state_dict only" and "dict with model_state")
ckpt = torch.load(ckpt_path, map_location=device)
if isinstance(ckpt, dict) and any(k in ckpt for k in ["model_state", "model_state_dict"]):
    state_dict = ckpt.get("model_state", ckpt.get("model_state_dict"))
else:
    state_dict = ckpt

model.load_state_dict(state_dict) # loading the model state
model.eval()

# generate function from prompt
@torch.no_grad()
def generate_from_prompt(
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 0,
) -> str:
    # encode prompt to token ids (shape: [1, T])
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # your GPT.generate expects (B, T) long tensor
    out_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature, # controls how random the next token selection is: lower = more predictable, higher = more creative
        top_k=top_k, # sampling to only the k most likely tokens, from its best guesses and preventing weird low-probability words.
    )[0].tolist()

    # decode to text
    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    return text

if __name__ == "__main__":
    prompt = "Once upon a time there was a dragon"
    story = generate_from_prompt(prompt, max_new_tokens=200, temperature=1, top_k=100)
    print(story)

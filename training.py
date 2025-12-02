import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import numpy as np
from model import *
from torch.optim.lr_scheduler import LambdaLR
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

# -----
block_size = 1024
batch_size = 32
eval_iters = 200
eval_interval = 2000

warmup_steps = 2000
max_lr = 2.5e-4
max_iters = 3
lr_decay_iters = max_iters # usually same



# -----

# data loader init
data_dir = "data/mini_openwebtext_3"
import numpy as np
import torch

data_dir = "data/mini_openwebtext_3"
block_size = 1024  # context length

train_data = np.memmap(f"{data_dir}/train.bin", dtype=np.uint16, mode="r")
val_data   = np.memmap(f"{data_dir}/val.bin", dtype=np.uint16, mode="r")


def get_batch(split, batch_size):
    if split == "train":
        data = train_data
    else:
        data = val_data

    ix = np.random.randint(0, len(data) - block_size, size=(batch_size,))
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix])

    # convert to torch tensors and proper dtype
    x = torch.from_numpy(x.astype(np.int64))  # or np.longlong
    y = torch.from_numpy(y.astype(np.int64))

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y

def lr_lambda(current_step):
    '''
    implementing LR increasing linearly from zero over the first 2000 updates
    '''
    # 1) linear warmup
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    # 2) cosine decay
    if current_step > lr_decay_iters:
        return 0.0
    decay_ratio = (current_step - warmup_steps) / float(max(1, lr_decay_iters - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return coeff

# loss function
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split, batch_size)
            logits, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


print('INITIALIZING MODEL')
# initial config
model_init = GPTConfig()

# initiate model
model = GPT(model_init)

model.to(device)

# intiialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, betas = (0.9, 0.95))
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# training loop
iter_num = 0

print('TRAINING STARTED')
while iter_num < max_iters:
    if iter_num % eval_interval == 0:
        loss = estimate_loss()
        print(f"step: {iter_num}, loss train: {loss['train']:.4f}, loss val: {loss['val']:.4f}")

    # sample batch
    x, y = get_batch("train", batch_size)

    # forward
    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    iter_num += 1

# optionally save
torch.save(model.state_dict(), "model_final.pt")
print("TRAINING DONE.")
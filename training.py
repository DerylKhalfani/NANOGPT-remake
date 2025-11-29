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

# initial config
model_init = GPTConfig()

# initiate model
model = GPT(model_init)

model.to(device)

# intiialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, betas = (0.9, 0.95))

print(model_init)
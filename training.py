import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from model import *
from torch.optim.lr_scheduler import LambdaLR
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

# -----
warmup_steps = 2000
max_lr = 2.5e-4
min_lr = 0.0
max_iters = 3
lr_decay_iters = max_iters # usually same

# -----

# data loader init



def lr_lambda(current_step):
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

def lr_lambda(current_step):
    '''
    implementing LR increasing linearly from zero over the first 2000 updates
    '''
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 0.5 * (1 + math.cos(math.pi * (current_step - warmup_steps) / (max_iters - warmup_steps)))


print(model_init)
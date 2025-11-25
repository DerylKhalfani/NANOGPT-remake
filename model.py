import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

"""
BASIC ARCHITECTURE FOR TRANSFORMER

Layer Norm
Causal Self Attention --> Block
Multi Layer Perceptron (FF Linear)
"""


"""
Original Hyperparams for GPT-2 Small
n_layer: 12
n_head: 12
n_embd: 768

"""

@dataclass
class GPTConfig:
    block_size: int = 512 # context length
    vocab_size:int = 50304 # GPT2-vocab_size, which is for BytePair Encoding
    n_layer: int = 8 # number of blocks
    n_head: int = 8 # number of attention heads
    n_embd: int = 512 # hidden dimension
    dropout: float = 0.1

class LayerNorm(nn.Module):
    pass

class MLP(nn.Module):
    pass

class SelfAttention(nn.Module):
    pass

class Block(nn.Module):
    pass

class GPT(nn.Module):
    pass






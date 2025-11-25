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
    n_embed: int = 512 # hidden dimension
    head_size:int = n_embed // n_head # size of head in attention mechanism
    dropout: float = 0.1
    bias:bool = True

class LayerNorm(nn.Module):
    pass

class MLP(nn.Module):
    pass

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        assert config.n_embed == config.n_head * config.head_size

        # key, query, value projections
        self.key = nn.Linear(config.n_embed, config.n_embed , bias=config.bias)
        self.query = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        self.value = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        # output projection
        self.projection = nn.Linear(config.n_embed, config.n_embed)

        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.head_size = config.head_size

        self.dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # using causal mask to make it become a decoder
            self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size)) # shape is for multi head

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension

        # calculating the query, key, values for all heads in batch
        # linear projections: (B, T, C)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # reshape to C dimension (n_head, head_size)
        q = q.view(B,T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # if self.flash:
        #     pass
        #
        # else:
        # computing attention weights
        att_wei = q @ k.transpose(-2, -1)
        att_wei = att_wei / self.head_size ** 0.5

        # applying masking so it cant see the future tokens only the previoues
        att_wei = att_wei.masked_fill(self.mask[:, :, :T, :T]==0, float('-inf'))

        # applying softmax for probabilities
        att_wei = F.softmax(att_wei, dim=-1) # row
        att_wei = self.dropout(att_wei)

        # last matmul to complete the attention formula
        y = att_wei @ v # (B, nh, T, hs)

        # merging heads back to: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # final linear projection + dropout
        y = self.projection(y)
        y = self.dropout(y)
        return y



class Block(nn.Module):
    pass

class GPT(nn.Module):
    pass






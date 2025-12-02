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

See the model architecture from gpt-1 paper since the architecture model is the same with gpt 2

"""

"""
What makes this different than nanoGPT:
    - Some of the code structure are more similar to Learning GPT from scratch
    - Smaller model due to compute limitation

"""

@dataclass
class GPTConfig:
    block_size: int = 1024 # context length
    vocab_size:int = 50304 # GPT2-vocab_size, which is for BytePair Encoding
    n_layer: int = 8 # number of blocks
    n_head: int = 8 # number of attention heads
    n_embed: int = 512 # hidden dimension
    head_size:int = n_embed // n_head # size of head in attention mechanism
    dropout: float = 0.1
    bias:bool = True

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            # shape input is (B,T, n_embed)
            nn.Linear(config.n_embed, 4 * config.n_embed, bias = config.bias),
            nn.GELU(), # gpt 1 uses GeLU
            nn.Linear(4 * config.n_embed, config.n_embed, bias = config.bias),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        x = self.net(x)

        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # make sure the embedding size can be evenly split into heads
        assert config.n_embed % config.n_head == 0
        # double check that head_size matches that split (sanity check)
        assert config.n_embed == config.n_head * config.head_size

        # Linear layer to produce keys Q, K, V from input x
        # input:  (B, T, n_embed)
        # output: (B, T, n_embed)
        self.key   = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        self.query = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)

        self.value = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)

        # Final linear that will be applied AFTER attention to mix heads together
        # input:  (B, T, n_embed)
        # output: (B, T, n_embed)
        self.projection = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)

        self.n_head   = config.n_head         # number of attention heads
        self.n_embed  = config.n_embed        # total embedding dimension (d_model)
        self.head_size = config.head_size     # per-head dimension (d_k)

        # dropout used on attention weights and on final projection
        self.dropout = nn.Dropout(config.dropout)

        # we detect if flash attention exists
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # create the causal mask used to prevent attention to future tokens
        # torch.ones(block_size, block_size) -> full matrix of 1s
        # tril(...) keeps only lower triangle (including diagonal), rest becomes 0
        # shape after view: (1, 1, block_size, block_size)
        # the leading 1,1 will broadcast over (B, n_head, T, T)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        # x: (B, T, C) = (batch, sequence length, embedding dim)
        B, T, C = x.size()

        # compute queries, keys and values for all tokens in the sequence
        # still shape: (B, T, C)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # reshape q, k, v to separate the heads
        # currently: (B, T, C) where C = n_head * head_size
        # view to:   (B, T, n_head, head_size)
        # transpose to: (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        # now q, k, v: (B, n_head, T, head_size) because we need to compute the attention scores

        # compute raw attention scores = Q K^T
        # q: (B, n_head, T, head_size)
        # k.transpose(-2, -1): (B, n_head, head_size, T)
        # result: (B, n_head, T, T)
        # each [i, j] is "how much token i attends to token j" for that head
        att_wei = q @ k.transpose(-2, -1)

        # formula from the paper divide by sqrt(head_size)
        att_wei = att_wei / (self.head_size ** 0.5)

        # apply causal mask so tokens cannot attend to future tokens
        # self.mask[:, :, :T, :T] -> (1, 1, T, T) broadcast over (B, n_head, T, T)
        # where mask == 0 (future), we set score to -inf so softmax -> 0 prob
        att_wei = att_wei.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        # convert scores to probabilities with softmax per row
        # For each query (per row) position i, att_wei[..., i, :] sums to 1
        att_wei = F.softmax(att_wei, dim=-1)

        #Dropout on attention weights (regularization)
        att_wei = self.dropout(att_wei)

        # weighted sum of values
        # att_wei: (B, n_head, T, T)
        # v:       (B, n_head, T, head_size)
        # result:  (B, n_head, T, head_size)
        # This is the "Att(Q, K, V) = softmax(QK^T / sqrt(d_k)) V" part
        y = att_wei @ v

        # merge heads back together Concatenate the head (paper 3.2.2)
        # currently: y is (B, n_head, T, head_size)
        # transpose to (B, T, n_head, head_size)
        # then view to (B, T, C) where C = n_head * head_size
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # final linear projection; mixes information across heads
        # shape remains (B, T, C)
        y = self.projection(y)

        # dropout on the output (regularization)
        y = self.dropout(y)

        # output has the same shape as input: (B, T, C)
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = CausalSelfAttention(config)
        self.MLP = MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        x = x + self.self_attention(self.ln1(x)) # pre layer norm like gpt 1 and residual connection residual connection
        x = x + self.MLP(self.ln2(x)) # residual connection

        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed) # embedding the token
        self.position_embedding = nn.Embedding(config.block_size, config.n_embed) # positional encoding
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)]) # how many layers (blocks)?

        # final layernorm
        self.ln_final = nn.LayerNorm(config.n_embed)

        # language modeling head
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias = False)

        # tying weight to help training
        self.lm_head.weight = self.token_embedding.weight

        # print number of weight
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        # helper to count parameters
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        '''
        Purpose:
        Take token ids as input
        Run them through the transformer
        Return logits (and optionally the loss)
        '''

        B,T = idx.size()

        token_embedding = self.token_embedding(idx) # (B,T,C)
        position_embedding = self.position_embedding(torch.arange(T, device=idx.device))

        x = token_embedding + position_embedding
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        # layer normalization final
        x = self.ln_final(x)

        if targets is None:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        else:
            logits = self.lm_head(x)
            B,T,V = logits.shape
            loss = F.cross_entropy(
                logits.view(B*T,V),
                targets.view(B*T)
            )

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature =1.0, top_k=None):
        '''

        '''
        for _ in range(max_new_tokens):
            # crop at block_size
            idx_cond = idx[:, -self.config.block_size:]

            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)

            # pluck logits at final step and scale bby desired temperature
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                pass

            # apply softmax to convert logits to probabilities to sample token
            probability = F.softmax(logits, dim=-1)

            # sample from distribution
            idx_next = torch.multinomial(probability, num_samples=1)

            # append new token to sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx











import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

"""
BASIC ARCHITECTURE FOR TRANSFORMER

Layer Norm
Causal Self Attention 
Multi Layer Perceptron (FF Linear)

Which Combines into a block
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
    - Smaller model due to hardware limitation
"""

@dataclass
class GPTConfig:
    """
    This class stores hyperparameters for nanoGPT(remake) model

    Used a config because its efficient (softcode)
    """
    block_size: int = 1024 # context length
    vocab_size:int = 50304 # GPT2-vocab_size, which is for BytePair Encoding
    n_layer: int = 8 # number of blocks
    n_head: int = 8 # number of attention heads
    n_embed: int = 512 # hidden dimension
    head_size:int = n_embed // n_head # size of head in attention mechanism
    dropout: float = 0.1
    bias:bool = True

class MLP(nn.Module):
    """
    This function defines the position wise feed forward network from the gpt 1 paper

    We need this because Attention mixes information across positions, the MLP mixes information
    within each position, non-linearly

    """

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            # shape input is (B,T, n_embed)
            # takes hidden vector size n_embed
            # first linear layer expands to 4 * n_embed
            nn.Linear(config.n_embed, 4 * config.n_embed, bias = config.bias),
            nn.GELU(), # gpt 1 uses GeLU

            # second linear layer compresses back to n_embed
            nn.Linear(4 * config.n_embed, config.n_embed, bias = config.bias),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        x = self.net(x)

        return x

class CausalSelfAttention(nn.Module):
    """
    Implementation of multi head self attention with a casual mask

    Core of transformer, each token looks at other tokens in the sequence
    to decide what to focus on
    """
    def __init__(self, config):
        super().__init__()
        # make sure the embedding size can be evenly split into heads
        assert config.n_embed % config.n_head == 0
        # double check that head_size matches that split (sanity check)
        assert config.n_embed == config.n_head * config.head_size

        # Linear layer to produce keys Q, K, V from input x
        # input:  (B, T, n_embed)
        # output: (B, T, n_embed)
        self.key   = nn.Linear(config.n_embed, config.n_embed, bias=config.bias) # Q: what this position is asking for
        self.query = nn.Linear(config.n_embed, config.n_embed, bias=config.bias) # K: What each position offers
        self.value = nn.Linear(config.n_embed, config.n_embed, bias=config.bias) # V: Actual content to be mixed

        # Final linear that will be applied AFTER attention to mix informatio accross heads together
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
        # we do this because token at position x can only look at positions before x

    def forward(self, x):
        # x: (B, T, C) = (batch size, sequence length, embedding dim)
        B, T, C = x.size()

        # compute queries, keys and values for all tokens in the sequence
        # still shape: (B, T, C)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # transform input to Q,K,V with three separate linear layers

        # reshape q, k, v to separate the heads
        # currently: (B, T, C) where C = n_head * head_size
        # view to:   (B, T, n_head, head_size)
        # transpose to: (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        # now q, k, v: (B, n_head, T, head_size) because we need to compute the attention scores

        # compute raw attention scores = Q K^T for each head
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
        # for each query (per row) position i, att_wei[..., i, :] sums to 1
        att_wei = F.softmax(att_wei, dim=-1)

        # dropout on attention weights (regularization)
        att_wei = self.dropout(att_wei)

        # weighted sum of values
        # att_wei: (B, n_head, T, T)
        # v:       (B, n_head, T, head_size)
        # result:  (B, n_head, T, head_size)
        # this is the "Att(Q, K, V) = softmax(QK^T / sqrt(d_k)) V" part
        y = att_wei @ v

        # merge heads back together Concatenate the head (paper 3.2.2) to a single tensor
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
    """
    Implements one transformer block LayerNorm -> Self Attetion -> Residual ->
    Self Attnetion -> MLP -> Residual

    Stacking blocks lets the model learn complex relationships across sequence
    """
    def __init__(self, config):
        super().__init__()
        self.self_attention = CausalSelfAttention(config)
        self.MLP = MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        """
        Works by:
        Pre-LayerNorm: normalize input before attention
        Self atteention returns some transformed features
        Add to original x (residual conection) (helps gradients, stabilizes training)
        Same pattern for MLP; normalize, apply MLP, then add residual

        """
        x = x + self.self_attention(self.ln1(x)) # pre layer norm like gpt 1 and residual connection
        x = x + self.MLP(self.ln2(x)) # residual connection

        return x

class GPT(nn.Module):
    """
    Overall GPT style language model

    Works by:
    token_embed -> maps toke IDs to vectors
    position_embded -> gives each position 0...block_size - 1 a learned vector
    dropout after addig embeddigs

    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed) # embedding the token
        self.position_embedding = nn.Embedding(config.block_size, config.n_embed) # positional encoding
        self.dropout = nn.Dropout(config.dropout)

        # stacking n_layer transformer blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)]) # how many layers (blocks)?

        # final layernorm
        self.ln_final = nn.LayerNorm(config.n_embed)

        # language modeling head
        # mapping hidden state -> logits over vocabulary
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias = False)

        # tying weight to help training
        # reusing token embeddig matrix as the output projection matrix, saves parameters and often helps performance
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

        # idx is the input
        B,T = idx.size()

        # getting token embeddings
        token_embedding = self.token_embedding(idx) # (B,T,C)

        # create [0,1,2,...,T-1] and look up positional embeddings
        position_embedding = self.position_embedding(torch.arange(T, device=idx.device))

        # add position info to tokens (how the model knows the order)
        x = token_embedding + position_embedding
        x = self.dropout(x)

        # run through all transformer blocks
        for block in self.blocks:
            x = block(x)

        # layer normalization final output
        x = self.ln_final(x)

        # if no targets (generation) compute logits for the last token in sequence
        if targets is None:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        # targets given -> compute logits for every position
        else:
            logits = self.lm_head(x)
            B,T,V = logits.shape
            # compute cross-entropy loss = standard token-level language modelling loss
            loss = F.cross_entropy(
                # flatten B,T,V to
                logits.view(B*T,V),
                # flatten targets B,T to
                targets.view(B*T)
            )

        return logits, loss

    # autoregressive sampling
    def generate(self, idx, max_new_tokens, temperature =1.0, top_k=None):
        '''
        Generate text by repeatedly sampling next tokens

        idx: starting tokens (prompt)
        max_new_tokens: maximum number of tokens to generate
        temperature: temperature for sampling, <1 for deterministic, > 1 more random
        top_k; sampling
        '''
        for _ in range(max_new_tokens):
            # trim context to last block_size
            idx_cond = idx[:, -self.config.block_size:]

            # forward the model to get the logits for all positions
            logits, _ = self(idx_cond)

            # pluck logits at final step and scale bby desired temperature
            # keep logits for last token
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                # keep only top_k logits, set the rest to -inf so probability ~ 0
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            # apply softmax to convert logits to probabilities to sample token from that categorical distribution
            probability = F.softmax(logits, dim=-1)

            # sample from distribution
            idx_next = torch.multinomial(probability, num_samples=1)

            # append new token to sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            # then repeat until max_new_tokens is reached

        return idx










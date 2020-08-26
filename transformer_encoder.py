import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_feed_forward import FeedForward
from transformer_multi_head_attention import MultiHeadAttention
from transformer_positional_encoder import PositionalEncoding


class EncoderLayer(nn.Module):
    # n_head: the number of multi-head attention
    # d_model : the dimensionality of embed
    # d_inner : the dimensionality of feed forward of inner layer
    def __init__(self, n_head, d_model, d_inner, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.feedforward = FeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, x, mask=None):  # x.shape: (batch, seqLen, embedSize)

        # enc_output.shape:(batch, seq, d_model)
        enc_output, enc_score = self.attn(x, x, x, mask=mask)
        enc_output = self.feedforward(enc_output)
        return enc_output, enc_score


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, pad_idx, dropout=0.1, max_seq_len=200):
        super(Encoder, self).__init__()
        self.src_word_emb = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(embed_size, n_position=max_seq_len)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList(
            [EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
             for _ in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    # x.shape:(batch, seqLen)  mask.shape:(batch, 1, seqLen)
    def forward(self, x, mask):

        enc_output = self.dropout(self.position_enc(self.src_word_emb(x)))
        enc_output = self.layer_norm(enc_output)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=mask)

        return enc_output








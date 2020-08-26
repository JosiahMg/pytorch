import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None): # q/k/v.shape: (batchSize, n_head, seqLen, dim)

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # attn.shape: (batchSize, n_head, q_seq, k_seq)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1)) # attn.shape: (batchSize, n_head, seqLen, seqLen)
        output = torch.matmul(attn, v) # output.shape: (batchSize, n_head, seqLen, dim)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head*d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head*d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head*d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q  # residual.shape:(batch, q_len, d_model)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: (batchSize, seqLen, n_head, dim)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: (batchSize, n_head, seqLen, d_k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None: # mask.shape: (batchSize, 1, seqLen)
            mask = mask.unsqueeze(1)   # For head axis broadcasting.  mask.shape: (batchSize, 1, 1, seqLen)

        # q.shape:(batch, n_head, len_q, d_v)
        # attn.shape: (batch, n_head, len_q, len_k)
        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q)) # q.shape: (batch, len_q, d_model)
        q += residual

        q = self.layer_norm(q)
        return q, attn



n_head = 8
d_model = 100
d_k = 50
d_v = 50
batch_size = 64
seq_len = 10

inputs = torch.randn(batch_size, seq_len, d_model)
print(inputs.shape)

atten = MultiHeadAttention(n_head, d_model, d_k, d_v)
outputs, score = atten(inputs, inputs, inputs)
print(outputs.shape)
print(score.shape)

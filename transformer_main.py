import torch
import torch.nn as nn
from transformer_encoder import Encoder
from transformer_decoder import Decoder


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class Transformer(nn.Module):
    def __init__(self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
                 d_word_vec=512, d_model=512, d_inner=2048, n_layers=6,
                 n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
                 trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True):
        super().__init__()

        assert d_model == d_word_vec

        self.encoder = Encoder(vocab_size=n_src_vocab, embed_size=d_word_vec, n_layers=n_layers,
                               n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner,
                               pad_idx=src_pad_idx, dropout=dropout, max_seq_len=n_position)

        self.decoder = Decoder(vocab_size=n_trg_vocab, embed_size=d_word_vec, n_layers=n_layers,
                               n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner,
                               pad_idx=trg_pad_idx, max_seqlen=n_position, dropout=dropout)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        if trg_emb_prj_weight_sharing:
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, src_seq, trg_seq):
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output = self.encoder(src_seq, src_mask)
        dec_output = self.decoder(trg_seq, trg_mask, enc_output, src_mask)

        # seq_logit.shape: (batch, seq_len, trg_vocab_size)
        seq_logit = self.trg_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))



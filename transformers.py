import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import copy
from typing import Optional

class Transformer(nn.Module):
    def __init__(self, d_model = 512, nheads = 8, num_encoder_layer = 6, 
                  num_decoder_layer = 6, dim_ffc = 2048, dropout = 0.1,
                 activation = 'relu', normalize_before = False):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.nheads = nheads
        
        encoder_layer = TransformerEncoderLayer(d_model, nheads, dim_ffc,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layer, encoder_norm)
        
        decoder_layer = TransformerDecoderLayer(d_model, nheads, dim_ffc,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layer, decoder_norm)
        self.position_encoder = PositionEmbeddingSine()

    def forward(self, x, tgt, pos = True, query_pos = None):
        if pos:
            pos = self.position_encoder(x)
        else: 
            pos = None
        memory = self.encoder(x, pos)
        hs = self.decoder(tgt, memory, pos, query_pos)
        return hs

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layer, norm = None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layer)
        self.num_layer = num_layer
        self.norm = norm

    def forward(self, x, pos):
        for layer in self.layers:
            x = layer(x, pos)
        if self.norm is not None:
            x = self.norm(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layer, norm = None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layer)
        self.num_layer = num_layer
        self.norm = norm

    def forward(self, x, memory, pos, query_pos):
        for layer in self.layers:
            x = layer(x, memory, pos, query_pos)
        if self.norm is not None:
            x = self.norm(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nheads, dim_ffc, dropout, activation, normalize_before):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nheads, dropout = dropout)
        self.linear1 = nn.Linear(d_model, dim_ffc)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ffc, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, x, pos):
        x2 = self.norm1(x)
        q = k = self.with_pos_embed(x2, pos)
        x2 = self.attention(q, k, value = x2)[0]
        x = x+self.dropout1(x2)
        x2 = self.activation(self.linear1(self.norm2(x)))
        x2 = self.linear2(self.dropout(x2))
        x = x+self.dropout2(x2)
        return x

    def forward_post(self, x, pos):
        q = k = self.with_pos_embed(x, pos)
        x2 = self.attention(q, k, value = x)[0]
        x = x+self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.activation(self.linear1(x))
        x2 = self.linear2(self.dropout(x2))
        x = x+self.dropout2(x2)
        x = self.norm2(x)
        return x

    def forward(self, x, pos):
        if self.normalize_before:
            return self.forward_pre(x, pos)
        else:
            return self.forward_post(x, pos)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nheads, dim_ffc, dropout, activation, normalize_before):
        super(TransformerDecoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nheads, dropout = dropout)
        self.multiattention = nn.MultiheadAttention(d_model, nheads, dropout = dropout)
        self.linear1 = nn.Linear(d_model, dim_ffc)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ffc, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, x, memory, pos, query_pos):
        x2 = self.norm1(x)
        q = k = self.with_pos_embed(x2, query_pos)
        x2 = self.attention(q, k, value = x2)[0]
        x = x+self.dropout1(x2)
        x2 = self.norm2(x)
        x2 = self.multiattention(query = self.with_pos_embed(x2, query_pos),
                                 key = self.with_pos_embed(memory, pos),
                                 value = memory)[0]
        x = x + self.dropout2(x2)
        x2 = self.activation(self.linear1(x))
        x2 = self.linear2(self.dropout(x2))
        x = x+self.dropout3(x2)
        return x

    def forward_post(self, x, memory, pos, query_pos):
        q = k = self.with_pos_embed(x, query_pos)
        x2 = self.attention(q, k, value = x)[0]
        x = x+self.dropout1(x2)
        x = self.norm1(x)
        x2 =  self.multiattention(query = self.with_pos_embed(x, query_pos),
                                  key = self.with_pos_embed(memory, pos),
                                  value = memory)[0]
        x = x+self.dropout2(x2)
        x = self.norm2(x)
        x2 = self.activation(self.linear1(x))
        x2 = self.linear2(self.dropout(x2))
        x = x+self.dropout3(x2)
        x = self.norm2(x)
        return x

    def forward(self, x, memory, pos, query_pos):
        if self.normalize_before:
            return self.forward_pre(x, memory, pos, query_pos)
        else:
            return self.forward_post(x, memory, pos, query_pos)


class PositionEmbeddingSine(nn.Module):
    def __init__(self, temperature=10000):
        super(PositionEmbeddingSine, self).__init__()
        self.temperature = temperature

    def forward(self, x):
        position_length, bs, d_model = x.shape
        position = np.array([[pos / self.temperature **(2 * (j // 2) / d_model) for j in range(d_model)]
        if pos != 0 else np.zeros(d_model) for pos in range(position_length)])
        position[1:, 0::2] = np.sin(position[1:, 0::2])
        position[1:, 1::2] = np.cos(position[1:, 1::2])
        position = torch.from_numpy(position).float()
        position = position.repeat(bs, 1, 1)
        return position.transpose(0,1).to(x.device)




def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    activation = activation.lower()
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def main():
    seq = 10 #sequence length
    bat = 16 #batch size
    dim = 512 #dimension
    x = torch.rand((seq, bat, dim))
    tgt = torch.rand((seq, bat, dim))
    transformer = Transformer()
    output = transformer(x, tgt)

if __name__ == '__main__':
    main()
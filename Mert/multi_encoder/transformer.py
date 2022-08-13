from typing import Optional
from torch.nn import *
from torch import Tensor


class TransformerEncoderLayer2(Module):
    __constants__ = ['batch_first']

    def __init__(
        self,
        d_model,
        nhead,
        S,
        L,
        dim_feedforward=2048,
        dropout=0.1,
        layer_norm_eps=1e-5,
        batch_first=False,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer2, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = ReLU()

        self.linear0 = Linear(S, L, **factory_kwargs)
        self.activation0 = ReLU()

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        src2 = self.self_attn(tgt, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]  # N*L*E
        src = self.activation0(self.linear0(src.permute(0, 2, 1)).permute(0, 2, 1))
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
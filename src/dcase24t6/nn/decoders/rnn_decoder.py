from typing import Optional

from torch import Tensor, nn

# class RNNDecoder(nn.Module):
#     def __init__(self, vocab_size: int, d_model: int, num_layers: int = 2):
#         super().__init__()

#         # Layers
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.rnn = nn.LSTM(d_model, d_model, num_layers, batch_first=True)
#         self.classifier = nn.Linear(d_model, vocab_size)


#     def forward(self, tgt: Tensor, memory: Tensor):
#         tgt_emb = self.embedding(tgt)
#         output, _ = self.rnn(tgt_emb)
#         return self.classifier(output)
class RNNDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        # pad_id: int,
        d_model: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.emb_layer = nn.Embedding(vocab_size, d_model)  # , padding_idx=pad_id)
        self.rnn = nn.LSTM(
            d_model, d_model, num_layers, dropout=dropout, batch_first=True
        )
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        frame_embs: Tensor,
        frame_embs_attn_mask: Optional[Tensor],
        frame_embs_pad_mask: Optional[Tensor],
        caps_in: Tensor,
        caps_in_attn_mask: Optional[Tensor],
        caps_in_pad_mask: Optional[Tensor],
    ) -> Tensor:
        embedded = self.emb_layer(caps_in)
        rnn_out, _ = self.rnn(embedded)
        logits = self.classifier(rnn_out)
        return logits

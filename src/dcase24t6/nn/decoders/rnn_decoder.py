from torch import nn, Tensor

class RNNDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int = 2):
        super().__init__()

        # Layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.LSTM(d_model, d_model, num_layers, batch_first=True)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, tgt: Tensor, memory: Tensor):
        tgt_emb = self.embedding(tgt)
        output, _ = self.rnn(tgt_emb)
        return self.classifier(output)
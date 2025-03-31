import torch.nn as nn
from layers import EncoderLayer, PositionalEncoding

class Transformers(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformers, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, attention_mask):
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, attention_mask.unsqueeze(1).unsqueeze(2))
        
        logits = self.fc(enc_output[:, 0, :])
        return logits

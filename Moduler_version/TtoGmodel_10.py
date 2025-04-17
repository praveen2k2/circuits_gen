import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TextToGraphTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_heads, num_layers, max_seq_len=512, dropout=0.1):
        super(TextToGraphTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.positional_encoding = SinusoidalPositionalEncoding(embedding_dim, max_seq_len)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Optional node classification head (multi-task)
        self.node_classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)  # Replace with actual number of classes if needed
        )

    def forward(self, input_seqs, seq_lengths=None, return_node_logits=False):
        """
        input_seqs: [B, S]
        seq_lengths: [B] (optional, not used here)
        return_node_logits: If True, returns node-level outputs for classification tasks
        """
        B, S = input_seqs.shape
        x = self.embedding(input_seqs)  # [B, S, D]
        x = self.positional_encoding(x)

        # Create attention mask: True for PAD tokens
        attention_mask = (input_seqs == 0)  # [B, S]
        x = self.transformer(x, src_key_padding_mask=attention_mask)  # [B, S, D]

        # Pairwise combinations for edges
        x_i = x.unsqueeze(2).expand(-1, -1, S, -1)  # [B, S, S, D]
        x_j = x.unsqueeze(1).expand(-1, S, -1, -1)  # [B, S, S, D]
        pairwise = torch.cat([x_i, x_j], dim=-1)   # [B, S, S, 2D]

        edge_logits = self.edge_mlp(pairwise).squeeze(-1)  # [B, S, S]
        edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2  # Make symmetric

        if return_node_logits:
            node_logits = self.node_classifier(x)  # [B, S, vocab_size] (or num_classes)
            return edge_logits, node_logits

        return edge_logits

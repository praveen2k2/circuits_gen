import torch
import torch.nn as nn

class TextToGraphTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_heads, num_layers, max_seq_len=512, dropout=0.1):
        super(TextToGraphTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.positional_encoding = nn.Parameter(torch.rand(1, max_seq_len, embedding_dim))

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
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_seqs, seq_lengths):
        """
        input_seqs: [B, S]
        seq_lengths: [B]
        """
        B, S = input_seqs.shape
        x = self.embedding(input_seqs)  # [B, S, D]
        x = x + self.positional_encoding[:, :S, :]

        # Create attention mask: True for PAD tokens
        attention_mask = (input_seqs == 0)  # [B, S]
        x = self.transformer(x, src_key_padding_mask=attention_mask)  # [B, S, D]

        # Pairwise combination
        x_i = x.unsqueeze(2).expand(-1, -1, S, -1)  # [B, S, S, D]
        x_j = x.unsqueeze(1).expand(-1, S, -1, -1)  # [B, S, S, D]
        pairwise = torch.cat([x_i, x_j], dim=-1)   # [B, S, S, 2D]

        edge_logits = self.edge_mlp(pairwise).squeeze(-1)  # [B, S, S]
        edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2

        return edge_logits

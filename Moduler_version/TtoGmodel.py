import torch
import torch.nn as nn

class TextToGraphTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_heads, num_layers, dropout=0.1):
        super(TextToGraphTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.rand(1, 512, embedding_dim))  # Example max sequence length

        # Transformer Encoder layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(embedding_dim, 1)  # Output: prediction for each pair (adjacency matrix entry)

    def forward(self, input_seqs, adj_mats, seq_lengths):
        # Embedding
        x = self.embedding(input_seqs)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]

        # Apply transformer encoder
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, embedding_dim)
        x = self.transformer(x)

        # Output: Predict the pairwise relationships (adjacency matrix values)
        adjacency_pred = self.fc(x)  # Shape: (seq_len, batch, 1)

        # Convert predictions to a matrix of size (batch, seq_len, seq_len)
        adjacency_matrix = adjacency_pred.squeeze(-1)  # Shape: (batch, seq_len, seq_len)
        adjacency_matrix = adjacency_matrix @ adjacency_matrix.transpose(-1, -2)  # Symmetric matrix

        return adjacency_matrix

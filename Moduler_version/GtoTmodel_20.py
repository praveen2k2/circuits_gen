import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphToTextTransformer(nn.Module):
    def __init__(self, graph_input_dim, vocab_size, embed_dim, num_heads, num_layers, dropout):
        super(GraphToTextTransformer, self).__init__()
        # Initialize layers (graph processing, transformer, etc.)
        # Example:
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, dropout=dropout),
            num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_dim, num_heads, dropout=dropout),
            num_layers
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, graph_input, tgt_input, tgt_mask=None, tgt_key_padding_mask=None):
        # Process the graph input (graph-to-text model)
        # Example:
        graph_embedding = self.graph_processing(graph_input)

        # Get target embeddings
        tgt_embedding = self.embedding(tgt_input)

        # Create source and target masks if not provided
        if tgt_mask is None:
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(tgt_input.device)
        
        # If we have a key padding mask, we apply it to the target sequence
        if tgt_key_padding_mask is not None:
            # Pass it to the transformer decoder
            decoder_output = self.decoder(
                tgt_embedding, 
                graph_embedding, 
                tgt_mask=tgt_mask, 
                memory_key_padding_mask=tgt_key_padding_mask
            )
        else:
            decoder_output = self.decoder(tgt_embedding, graph_embedding, tgt_mask=tgt_mask)

        output = self.fc_out(decoder_output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, embed_dim)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor (batch_size, seq_len, embed_dim)
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.dtype)
        return x

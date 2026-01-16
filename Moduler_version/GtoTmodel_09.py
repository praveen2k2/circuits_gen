import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphToTextTransformer(nn.Module):
    def __init__(self, graph_input_dim, text_vocab_size, embed_dim, num_heads, num_layers, dropout=0.1):
        super(GraphToTextTransformer, self).__init__()
        self.embed_dim = embed_dim
        
        # Encoder embedding layer for graph data
        self.encoder_embedding = nn.Linear(graph_input_dim, embed_dim)
        
        # Decoder embedding layer for text data
        self.decoder_embedding = nn.Embedding(text_vocab_size, embed_dim)
        
        # Positional encoding for both encoder and decoder
        self.positional_encoding = PositionalEncoding(embed_dim)
        
        # Transformer model
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout
        )
        
        # Output layer to generate text predictions
        self.output_layer = nn.Linear(embed_dim, text_vocab_size)

    def forward(self, graph_data, text_input, src_mask=None, tgt_mask=None):
        """
        Args:
            graph_data: Tensor of shape (batch_size, seq_len, graph_input_dim)
            text_input: Tensor of shape (batch_size, tgt_seq_len)
            src_mask: Optional mask for the encoder input
            tgt_mask: Optional mask for the decoder input
         Returns:
            Tensor of shape (batch_size, tgt_seq_len, text_vocab_size)
        """
        # Create a causal mask for the target sequence (if not provided)
        if tgt_mask is None:
            tgt_seq_len = text_input.size(1)
            tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1).bool().to(graph_data.device)
        
        # Encode graph data
        graph_encoded = self.encoder_embedding(graph_data)  # (batch_size, seq_len, embed_dim)
        graph_encoded = self.positional_encoding(graph_encoded)  # Add positional encoding
        
        graph_encoded = graph_encoded.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        
        # Embed text input and add positional encoding
        text_embedded = self.decoder_embedding(text_input)  # (batch_size, tgt_seq_len, embed_dim)
        text_embedded = self.positional_encoding(text_embedded)  # Add positional encoding
        
        text_embedded = text_embedded.permute(1, 0, 2)  # (tgt_seq_len, batch_size, embed_dim)
        
        # Pass through transformer
        transformer_output = self.transformer(
            src=graph_encoded,
            tgt=text_embedded,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )  # (tgt_seq_len, batch_size, embed_dim)
        
        # Map to text vocabulary
        output = self.output_layer(transformer_output)  # (tgt_seq_len, batch_size, text_vocab_size)
 
        final_output = output.permute(1, 0, 2)  # (batch_size, tgt_seq_len, text_vocab_size)

        return final_output

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create a tensor to hold the positional encodings
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices: sine
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices: cosine
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to the input tensor `x`.
        Args:
            x: Tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            Tensor with positional encoding added
        """
        # x.shape is (batch_size, seq_len, embed_dim)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

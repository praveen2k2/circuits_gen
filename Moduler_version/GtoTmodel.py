import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphToTextTransformer(nn.Module):
    def __init__(self, graph_input_dim, text_vocab_size, embed_dim, num_heads, num_layers, dropout=0.1):
        super(GraphToTextTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.text_vocab_size = text_vocab_size
        
        # Encoder embedding for graph data
        self.encoder_embedding = nn.Linear(graph_input_dim, embed_dim)
        
        # Decoder embedding for text input
        self.decoder_embedding = nn.Embedding(text_vocab_size, embed_dim)
        
        # Transformer model
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout
        )
        
        # Output layer with attention mechanism
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, text_vocab_size)
        )

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
        # Encode graph data
        print("Input graph_data shape:", graph_data.shape)
        graph_encoded = self.encoder_embedding(graph_data)  # (batch_size, seq_len, embed_dim)
        print("graph_encoded shape:", graph_encoded.shape)
        graph_encoded = graph_encoded.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        print("graph_encoded permuted shape:", graph_encoded.shape)

        # Embed text input
        text_embedded = self.decoder_embedding(text_input)  # (batch_size, tgt_seq_len, embed_dim)
        print("text_embedded shape:", text_embedded.shape)
        text_embedded = text_embedded.permute(1, 0, 2)  # (tgt_seq_len, batch_size, embed_dim)
        print("text_embedded permuted shape:", text_embedded.shape)

        # Pass through transformer
        transformer_output = self.transformer(
            src=graph_encoded,
            tgt=text_embedded,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )  # (tgt_seq_len, batch_size, embed_dim)
        print("transformer_output shape:", transformer_output.shape)
        
        # Permute back to original shape for output layer
        transformer_output = transformer_output.permute(1, 0, 2)  # (batch_size, tgt_seq_len, embed_dim)
        print("output shape:", transformer_output.shape)

        # Apply output layer to each timestep
        output = self.output_layer(transformer_output)  # (batch_size, tgt_seq_len, text_vocab_size)
        print("final_output shape:", output.shape)

        return output
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Embedding layers for encoder and decoder inputs
        self.encoder_embedding = nn.Linear(input_dim, embed_dim)
        self.decoder_embedding = nn.Linear(input_dim, embed_dim)
        self.component_embedding = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, embed_dim))  # Adjust max length as needed
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout
        )
        
        # Output layer
        self.output_layer = nn.Linear(embed_dim, input_dim)

    def forward(self, encoder_input, decoder_input, component_input):
        # Encoder input embedding
        encoder_embedded = self.encoder_embedding(encoder_input) + self.positional_encoding[:, :encoder_input.size(1), :]
        encoder_embedded = encoder_embedded.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        
        # Decoder input embedding
        decoder_embedded = self.decoder_embedding(decoder_input) + self.positional_encoding[:, :decoder_input.size(1), :]
        component_embedded = self.component_embedding(component_input)
        combined_decoder_input = decoder_embedded + component_embedded
        combined_decoder_input = combined_decoder_input.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        
        # Transformer forward pass
        transformer_output = self.transformer(
            src=encoder_embedded,
            tgt=combined_decoder_input
        )
        
        # Final output
        transformer_output = transformer_output.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
        output = self.output_layer(transformer_output)
        return output


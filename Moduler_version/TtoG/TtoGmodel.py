import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphToTextTransformer(nn.Module):
    def __init__(self, graph_input_dim, text_vocab_size, embed_dim, num_heads, num_layers, dropout=0.1):
        super(GraphToTextTransformer, self).__init__()
        self.embed_dim = embed_dim
        
        self.encoder_embedding = nn.Embedding(text_vocab_size, embed_dim)
        
        self.decoder_embedding = nn.Linear(graph_input_dim, embed_dim)
        
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout
        )
        
        self.output_layer = nn.Linear(embed_dim, graph_input_dim)

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
        # Create a causal mask for the target sequence
        if tgt_mask is None:
            tgt_seq_len = text_input.size(1)
            tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1).bool().to(graph_data.device)
        # Encode graph data
        graph_encoded = self.encoder_embedding(graph_data)  # (batch_size, seq_len, embed_dim)
        
        graph_encoded = graph_encoded.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        
        # Embed text input
        text_embedded = self.decoder_embedding(text_input) # (batch_size, tgt_seq_len, embed_dim)
        
        text_embedded = text_embedded.permute(1, 0, 2)  # (tgt_seq_len, batch_size, embed_dim)
        
        # Pass through transformer
        transformer_output = self.transformer(
            src=text_embedded,
            tgt=graph_encoded,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )  # (tgt_seq_len, batch_size, embed_dim)
        
        # Map to text vocabulary
        output = self.output_layer(transformer_output)  # (tgt_seq_len, batch_size, text_vocab_size)
 
        final_output = output.permute(1, 0, 2)  # (batch_size, tgt_seq_len, text_vocab_size)

        return final_output

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class EdgePredictionModel(nn.Module):
    def __init__(self, in_node_features, in_edge_features, num_classes):
        super(EdgePredictionModel, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=in_node_features, nhead=4), num_layers=2
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=in_edge_features, nhead=4), num_layers=2
        )
        self.fc = nn.Linear(in_edge_features, num_classes)  # Output layer for edge predictions

    def forward(self, graph_encoding, component_encoding):
        # graph_encoding: (seq_len, batch_size, in_node_features)
        # component_encoding: (seq_len, batch_size, in_edge_features)

        # Pass through the encoder
        encoded_graph = self.encoder(graph_encoding)

        # Pass through the decoder
        decoded_edges = self.decoder(component_encoding, encoded_graph)

        # Generate edge predictions
        edge_predictions = self.fc(decoded_edges)

        return edge_predictions
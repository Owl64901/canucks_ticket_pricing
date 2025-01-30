import torch
import torch.nn as nn
import numpy as np

class TicketSalesTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, output_dim, max_length):
        super(TicketSalesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = self.create_positional_encoding(max_length, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, output_dim)
        # Activation function ensuring non-negativity
        self.non_negative_activation = nn.Softplus()

        # Correct initialization of BatchNorm1d
        self.batch_norm = nn.BatchNorm1d(max_length)
        self.dropout = nn.Dropout(0.2)

    def create_positional_encoding(self, max_length, embed_dim):
        # Create positional encoding based on given max_length and embed_dim
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_length, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        # Forward pass through the transformer model
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :].to(x.device)
        x = self.batch_norm(x)
        x = self.transformer(x)
        x = self.dropout(x)
        x = self.fc_out(x[:, -1, :])
        x = self.non_negative_activation(x)
        return x
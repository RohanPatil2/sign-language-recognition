from dataclasses import asdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AttentionPooling(nn.Module):
    """Multi-head self-attention pooling with context query"""
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_size))
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, lengths=None):
        # x: (batch_size, seq_len, hidden_size)
        query = self.query.unsqueeze(0).repeat(x.size(0), 1, 1)
        attn_output, _ = self.attention(query, x, x)
        return self.layer_norm(attn_output.squeeze(1))

class HighwayLSTM(nn.Module):
    """Bidirectional LSTM with highway connections and attention pooling"""
    def __init__(self, config, n_classes=50):
        super().__init__()
        config_dict = asdict(config)
        self.config = config
        
        # Enhanced LSTM configuration
        self.lstm = nn.LSTM(**config_dict)
        self.lstm_dropout = nn.Dropout(config.dropout)
        
        # Attention pooling
        self.attention = AttentionPooling(
            hidden_size=config.hidden_size * (2 if config.bidirectional else 1),
            num_heads=config.attention_heads
        )
        
        # Highway network
        self.highway = nn.Linear(config.hidden_size, config.hidden_size)
        self.gate = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Projection head
        in_features = config.hidden_size * (2 if config.bidirectional else 1)
        self.projection = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, config.projection_size),
            nn.SiLU(),
            nn.Dropout(config.dropout)
        )
        
        # Classification head
        self.classifier = nn.Linear(config.projection_size, n_classes)
        
        # Initialize parameters
        self.init_weights()

    def init_weights(self):
        """Orthogonal initialization for LSTM parameters"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # Set forget gate bias to 1
                if 'bias_ih' in name:
                    param.data[param.size(0)//4:param.size(0)//2].fill_(1.0)
                if 'bias_hh' in name:
                    param.data[param.size(0)//4:param.size(0)//2].fill_(1.0)

        nn.init.xavier_normal_(self.highway.weight)
        nn.init.xavier_normal_(self.gate.weight)

    def forward(self, x, lengths=None):
        # Handle packed sequences
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x, _ = pad_packed_sequence(x, batch_first=True)
        
        # BiLSTM processing
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # Highway network
        highway_transform = F.relu(self.highway(lstm_out))
        gate = torch.sigmoid(self.gate(lstm_out))
        highway_out = gate * highway_transform + (1 - gate) * lstm_out
        
        # Attention pooling
        attention_out = self.attention(highway_out, lengths)
        
        # Multi-pooling strategy
        max_pool = torch.max(highway_out, dim=1).values
        mean_pool = torch.mean(highway_out, dim=1)
        pooled = torch.cat([attention_out, max_pool, mean_pool], dim=1)
        
        # Projection and classification
        projected = self.projection(pooled)
        return self.classifier(projected)

class AdvancedLSTMConfig:
    def __init__(self, input_size, hidden_size, num_layers=2, 
                 bidirectional=True, dropout=0.3, attention_heads=4,
                 projection_size=512):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.attention_heads = attention_heads
        self.projection_size = projection_size
"""
PyTorch Intent Classification Model
Optimized for elderly speech patterns and on-device inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntentClassifier(nn.Module):
    """
    Lightweight intent classification model for on-device inference.
    Uses BiLSTM + Attention for capturing context in elderly speech.
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 num_intents=10, dropout=0.3, num_layers=2):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension for LSTM
            num_intents: Number of intent classes
            dropout: Dropout rate for regularization
            num_layers: Number of LSTM layers
        """
        super(IntentClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_intents)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
    def attention_net(self, lstm_output):
        """
        Attention mechanism to focus on important words
        """
        # lstm_output shape: (batch_size, seq_len, hidden_dim * 2)
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        # attention_weights shape: (batch_size, seq_len, 1)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_output, dim=1)
        # context shape: (batch_size, hidden_dim * 2)
        
        return context, attention_weights
        
    def forward(self, x, lengths=None):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len)
            lengths: Actual lengths of sequences (for packing)
            
        Returns:
            logits: Intent classification logits (batch_size, num_intents)
            attention_weights: Attention weights for interpretability
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Pack sequence if lengths provided (for efficiency)
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
        
        # BiLSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Unpack if packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        
        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Apply attention
        context, attention_weights = self.attention_net(lstm_out)
        
        # Fully connected layers
        out = F.relu(self.fc1(context))
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits, attention_weights


class FastIntentClassifier(nn.Module):
    """
    Ultra-lightweight model for faster inference on low-power devices.
    Uses CNN instead of LSTM for speed.
    """
    
    def __init__(self, vocab_size, embedding_dim=64, num_filters=128, 
                 filter_sizes=[2, 3, 4], num_intents=10, dropout=0.3):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            num_filters: Number of filters per filter size
            filter_sizes: List of filter sizes (n-grams)
            num_intents: Number of intent classes
            dropout: Dropout rate
        """
        super(FastIntentClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Multiple CNN layers with different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_intents)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len)
            
        Returns:
            logits: Intent classification logits (batch_size, num_intents)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Transpose for Conv1d (expects channels first)
        embedded = embedded.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, new_seq_len)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # (batch_size, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))  # (batch_size, num_filters)
        
        # Concatenate all conv outputs
        cat = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))
        
        # Dropout and FC
        cat = self.dropout(cat)
        logits = self.fc(cat)
        
        return logits


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the models
    vocab_size = 5000
    num_intents = 10
    batch_size = 4
    seq_len = 20
    
    # Test BiLSTM model
    print("Testing BiLSTM Intent Classifier...")
    model = IntentClassifier(vocab_size=vocab_size, num_intents=num_intents)
    print(f"Model parameters: {count_parameters(model):,}")
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    lengths = torch.tensor([15, 20, 10, 18])
    
    logits, attention = model(x, lengths)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Attention shape: {attention.shape}")
    
    # Test Fast CNN model
    print("\nTesting Fast CNN Intent Classifier...")
    fast_model = FastIntentClassifier(vocab_size=vocab_size, num_intents=num_intents)
    print(f"Model parameters: {count_parameters(fast_model):,}")
    
    logits = fast_model(x)
    print(f"Output shape: {logits.shape}")
    
    print("\nModels initialized successfully!")
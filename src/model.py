from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor

class Encoder(nn.Module):
    """A two-layer LSTM encoder that compresses a sequence of sensor readings into a context vector.

    Processes a batch of time series sequences and returns the final hidden and cell states of the LSTM, forming the latent representation.

    Args:
        n_features: Number of input features per time step.
        sequence_length: Length of the input window in time steps.
        embedding_dim: Dimensionality of the latent space.
    """
    
    def __init__(self, n_features: int, sequence_length: int, embedding_dim: int) -> None:
        super(Encoder, self).__init__()
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=2,
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode a batch of sequences.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_features).

        Returns:
            A tuple containing:
                - hidden: Final hidden state tensor of shape (num_layers, batch_size, embedding_dim).
                - cell: Final cell state tensor of shape (num_layers, batch_size, embedding_dim).
        """
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    """A two-layer LSTM decoder that reconstructs sequences from a context vector.

    Takes the final hidden and cell states from the encoder and generates the output sequence by unrolling and LSTM over a dummy zero input, then mapping each time step back to the original feature space.

    Args:
        n_features: Number of output features per time step.
        sequence_length: Length of the output window in time steps.
        embedding_dim: Dimensionality of the latent space.
    """
    
    def __init__(self, n_features: int, sequence_length: int, embedding_dim: int) -> None:
        super(Decoder, self).__init__()
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=2,
            batch_first=True,
        )
        self.fc = nn.Linear(embedding_dim, n_features) # fully connected layer to map to output features

    def forward(self, hidden: Tensor, cell: Tensor) -> Tensor:
        """Decode a batch of sequences from the given context (hidden and cell states).

        Args:
            hidden: Initial hidden state tensor of shape (num_layers, batch_size, embedding_dim).
            cell: Initial cell state tensor of shape (num_layers, batch_size, embedding_dim).

        Returns:
            Reconstructed output tensor of shape (batch_size, sequence_length, n_features).
        """
        
        # Initialize decoder input with zeros
        batch_size = hidden.size(1)
        
        # Dummy input of zeros, one time step per row
        decoder_input = torch.zeros(
            batch_size,
            self.sequence_length,
            self.n_features,
            device=hidden.device,
        )
        
        reconstructed, _ = self.lstm(decoder_input, (hidden, cell))
        reconstructed = self.fc(reconstructed)
        return reconstructed
    
class Autoencoder(nn.Module):
    """Seq2Seq autoencoder for turbofan sensor sequences.

    The model encodes a sequence of sensor readings into a latent context vector
    and decodes it back, learning to reconstruct normal operating behavior.
    """
    
    def __init__(self, n_features: int, sequence_length: int, embedding_dim: int) -> None:
        super().__init__()
        self.encoder = Encoder(n_features, sequence_length, embedding_dim)
        self.decoder = Decoder(n_features, sequence_length, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_features).

        Returns:
            Reconstructed output tensor of shape (batch_size, sequence_length, n_features).
        """
        hidden, cell = self.encoder(x)
        reconstructed = self.decoder(hidden, cell)
        return reconstructed
    

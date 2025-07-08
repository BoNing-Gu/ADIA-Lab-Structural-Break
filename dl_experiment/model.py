import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import config

class SiameseLSTM(nn.Module):
    def __init__(self):
        super(SiameseLSTM, self).__init__()
        
        # LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=config.INPUT_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.N_LAYERS,
            bidirectional=config.BIDIRECTIONAL,
            dropout=config.DROPOUT if config.N_LAYERS > 1 else 0,
            batch_first=True
        )
        
        # Calculate MLP input dimension
        lstm_output_dim = config.HIDDEN_DIM * 2 if config.BIDIRECTIONAL else config.HIDDEN_DIM
        
        # MLP Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, config.OUTPUT_DIM),
            nn.Sigmoid()
        )

    def forward_one(self, packed_x):
        # packed_x is a tuple (padded_sequence, lengths)
        x_padded, x_lengths = packed_x
        
        # Pack the padded batch of sequences
        # We need to enforce_sorted=False because the dataloader doesn't guarantee sorted order
        packed_input = pack_padded_sequence(x_padded, x_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Pass packed batch to LSTM
        _, (hidden, _) = self.lstm(packed_input)
        
        # Extract the final hidden state
        # hidden shape: (num_layers * num_directions, batch, hidden_size)
        if self.lstm.bidirectional:
            # Concatenate the final hidden states of the last layer from both directions
            # The last forward layer is at index -2, the last backward layer is at index -1
            forward_hidden = hidden[-2,:,:]
            backward_hidden = hidden[-1,:,:]
            embedding = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            # The last layer's hidden state
            embedding = hidden[-1,:,:]
            
        return embedding

    def forward(self, phase1_packed, phase2_packed):
        # Get embeddings for both phases
        embedding1 = self.forward_one(phase1_packed)
        embedding2 = self.forward_one(phase2_packed)
        
        # Concatenate embeddings
        combined_embedding = torch.cat((embedding1, embedding2), dim=1)
        
        # Pass through classifier
        output = self.classifier(combined_embedding)
        
        return output.squeeze()

if __name__ == '__main__':
    # Test the model with dummy data using the new structure
    model = SiameseLSTM().to(config.DEVICE)
    print(model)

    # Dummy input
    batch_size = 4 # smaller for testing
    seq_len1 = [150, 120, 100, 80]
    seq_len2 = [200, 180, 150, 130]

    phase1_dummy = [torch.randn(l, config.INPUT_DIM) for l in seq_len1]
    phase2_dummy = [torch.randn(l, config.INPUT_DIM) for l in seq_len2]

    phase1_padded = nn.utils.rnn.pad_sequence(phase1_dummy, batch_first=True).to(config.DEVICE)
    phase2_padded = nn.utils.rnn.pad_sequence(phase2_dummy, batch_first=True).to(config.DEVICE)

    phase1_lengths = torch.tensor(seq_len1).to(config.DEVICE)
    phase2_lengths = torch.tensor(seq_len2).to(config.DEVICE)

    # Forward pass
    output = model((phase1_padded, phase1_lengths), (phase2_padded, phase2_lengths))
    
    print("\nOutput shape:", output.shape)
    assert output.shape == (batch_size,)
    print("Model test passed!") 
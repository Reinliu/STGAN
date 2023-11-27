import torch
import torch.nn as nn
freq_res = 64

class SpectrogramGenerator(nn.Module):
    def __init__(self, n_classes, seq_len, latent_dim=100, loudness_dim=1, hidden_dim=512, n_layers=3):
        super(SpectrogramGenerator, self).__init__()
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        # Embedding for labels
        self.label_embedding = nn.Sequential(
            nn.Embedding(self.n_classes, self.n_classes*10),
            nn.Linear(self.n_classes*10, self.seq_len)
        )
        label_dim = 1
        # If using a single latent vector, expand it to match the sequence length
        self.fc_init = nn.Linear(latent_dim + loudness_dim + label_dim, hidden_dim)

        # Define the RNN layer(s)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True)

        # Define the output layer
        self.fc_out = nn.Linear(hidden_dim, freq_res)

    def forward(self, z, loudness, labels):
        seq_len = loudness.size(1)  # This should be 400
        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        # Expand the latent space to match the sequence length and concatenate with loudness
        label_emb = self.label_embedding(labels)
        label_emb = label_emb.view(labels.size(0), seq_len, 1)
        rnn_input = torch.cat((z, loudness, label_emb), dim=2)
        # Process the initial input to match the RNN input features
        rnn_input = self.fc_init(rnn_input)

        # Get the RNN output for the whole sequence
        rnn_out, _ = self.rnn(rnn_input)

        # Process the RNN outputs to the final spectrogram shape
        output_seq = self.fc_out(rnn_out.contiguous().view(-1, self.hidden_dim))
        spectrogram = output_seq.view(-1, 1, seq_len, freq_res)

        return spectrogram


class Discriminator(nn.Module):
    def __init__(self, n_classes):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Sequential(
            nn.Embedding(n_classes, n_classes*10),
            nn.Linear(n_classes*10, 25600)
        )
        self.model = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(25600, 1)
        )

    def forward(self, img, labels):
        # Expand label embedding to same size as image
        seq_len = img.size(2)
        labels = self.label_embedding(labels)
        labels = labels.view(labels.size(0), 1, seq_len, freq_res)
        # Concatenate label and image
        d_in = torch.cat((img, labels), 1) # Concatenate along channel dimension
        output = self.model(d_in)
        return output
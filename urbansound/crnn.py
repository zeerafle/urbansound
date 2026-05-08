import torch
from torch import nn
from torchsummary import summary


class CRNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # GRU layer
        # The input_size is channels (128) * freq_bins (5) = 640
        self.gru = nn.GRU(input_size=128 * 5, hidden_size=64, batch_first=True)

        # Linear layer for classification into 10 classes
        self.linear = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        # Extract features using CNN
        # input_data shape: (batch_size, 1, 64, 44)
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # The shape of x is now (batch_size, channels=128, freq=5, time=4)
        batch_size, channels, freq, time = x.size()

        # Flatten the channel and frequency dimensions together
        # Resulting shape: (batch_size, channels * freq, time)
        x = x.view(batch_size, channels * freq, time)

        # Transpose so time is the sequence dimension: (batch_size, seq_len=time, features)
        # Resulting shape: (batch_size, 4, 640)
        x = x.transpose(1, 2)

        # Pass through the GRU
        # rnn_out shape: (batch_size, time, hidden_size)
        rnn_out, _ = self.gru(x)

        # Extract the hidden state from the last time step
        # Output shape: (batch_size, hidden_size)
        last_time_step = rnn_out[:, -1, :]

        # Pass through linear and softmax
        logits = self.linear(last_time_step)
        predictions = self.softmax(logits)

        return predictions


if __name__ == "__main__":
    crnn = CRNNNetwork()
    # Detect if an NVIDIA GPU is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Move the model to the detected device
    crnn = crnn.to(device)

    # Execute summary
    summary(crnn, (1, 64, 44))

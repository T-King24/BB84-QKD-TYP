import torch
import torch.nn as nn


class QKDGatekeeperLSTM(nn.Module):
    """
    Long Short-Term Memory (LSTM) neural network that detects eavesdropping
    by looking at the telemetry of a BB84 link over a window of timesteps
    (error rates, detection rates, dark/check-slot clicks, timing, watchdog, ...)
    """
    def __init__(self, input_size=11, hidden_dim=64, num_layers=2, dropout=0.2):
        super(QKDGatekeeperLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,  # features per timestamp (len(FEATURES) in data_gen.py)
            hidden_size=hidden_dim,  # nodes to capture temporal patterns
            num_layers=num_layers,  # number of lstm layers for deeper feature extraction
            batch_first=True,  # makes sure input tensors are in right format
            dropout=dropout if num_layers > 1 else 0  # regularisation to avoid overfitting
        )

        # linear layer to map hidden state to singular output
        self.fc = nn.Linear(hidden_dim, 1)

        # sigmoid activation to turn output into probability
        self.sigmoid = nn.Sigmoid()

    # defines the forward pass
    def forward(self, x):

        # pass the input through the layers where lstm_out has the output features
        # for each time step
        lstm_out, hidden_state = self.lstm(x)

        # extract last output from last timestep
        last_timestep_out = lstm_out[:, -1, :]
        out = self.fc(last_timestep_out)

        # convert output to probability
        prob = self.sigmoid(out)

        # return so shape is (batchsize,)
        return prob.squeeze(-1)


if __name__ == '__main__':

    model = QKDGatekeeperLSTM()
    dummyX = torch.randn(64, 50, 11)
    output = model(dummyX)
    print(output.shape)
    print(output[:5])

import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# make data_gen / bb84 importable when this file is run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_gen import FEATURES, MAIN_CSV, MAIN_SEED, load_or_generate  # noqa: E402

DEFAULT_CSV = MAIN_CSV


def ensure_dataset(csv_path=DEFAULT_CSV, seed=MAIN_SEED):
    """
    Return the telemetry dataset. For the main dataset this always matches the current
    settings in data_gen.py (simulated with the BB84 link if needed, otherwise loaded
    from the cache). Any other csv_path is simply read.
    """
    if os.path.normpath(csv_path) == os.path.normpath(MAIN_CSV):
        df = load_or_generate(seed)
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        df.to_csv(csv_path, index=False)
        return df
    return pd.read_csv(csv_path)


class QKDSequenceDataset(Dataset):
    # dataset for QKD telemetry
    # uses look back sequences that will be used for LSTM network
    def __init__(self, df, window_size=50, scaler=None, is_train=True, features=FEATURES):
        # number of timesteps used for prediction
        self.window_size = window_size
        self.features = list(features)

        # extract features and ground truths
        X = df[self.features].to_numpy(dtype=float)
        y = df["attack_label"].to_numpy()

        if is_train:
            # fit training set to new standard scaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            # if test set use scaler from the training set
            self.scaler = scaler
            X_scaled = self.scaler.transform(X)

        self.X_seq = []
        self.y_seq = []

        # slide window across dataset
        for i in range(0, len(X_scaled) - window_size):
            self.X_seq.append(X_scaled[i: i + window_size])
            # last timestep assigned a label as target to predict
            self.y_seq.append(y[i + window_size - 1])

        # metadata of the last timestep of each window (for per-attack evaluation)
        last = np.arange(window_size - 1, len(X_scaled) - 1)
        self.attack_type = df["attack_type"].to_numpy()[last]
        self.is_stealthy = df["is_stealthy"].to_numpy().astype(bool)[last]
        self.intensity = df["attack_intensity"].to_numpy()[last]

        # convert lists to pytorch tensors
        self.X_seq = torch.tensor(np.array(self.X_seq), dtype=torch.float32)
        self.y_seq = torch.tensor(np.array(self.y_seq), dtype=torch.float32)

    def __len__(self):
        # return number of sequence windows used
        return len(self.X_seq)

    def __getitem__(self, index):
        # returns a (sequence, label) pair
        return self.X_seq[index], self.y_seq[index]


def get_qkd_dataloaders(csv_path=DEFAULT_CSV,
                        window_size=50,
                        batch_size=64,
                        train_split=0.8,
                        seed=42,
                        features=FEATURES):

    # load telemetry CSV (simulated with the BB84 link if missing)
    df = ensure_dataset(csv_path)

    # split into 80% training 20% test and made sure to split sequentially
    # to keep temporal order
    split_idx = int(len(df) * train_split)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    # creates training dataset
    train_dataset = QKDSequenceDataset(df=train_df, window_size=window_size,
                                       is_train=True, features=features)

    # creates test dataset
    test_dataset = QKDSequenceDataset(df=test_df, window_size=window_size,
                                      scaler=train_dataset.scaler, is_train=False,
                                      features=features)

    # create the Pytorch dataloaders, which batch the data and move it to the models
    # shuffle = true to prevent model from learning exact sequence from the training set
    # and false for test set to keep chronological order for evaluation
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("DataLoaders successfully created!")
    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    print(f"Input tensor shape per batch: ({batch_size}, {window_size}, {len(features)})")

    # return dataloaders and scaler
    return train_loader, test_loader, train_dataset.scaler


if __name__ == '__main__':
    train_loader, test_loader, scaler = get_qkd_dataloaders()
    for X_batch, y_batch in train_loader:
        print("\nBatch Verification:")
        print(f"X_batch shape: {X_batch.shape}")
        print(f"y_batch shape: {y_batch.shape}")
        break

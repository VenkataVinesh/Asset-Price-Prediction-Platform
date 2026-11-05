"""Stacked LSTM that predicts the next log return from a 30-step window of
[log return, RSI/100, rolling volatility].

It is actually trained, on a chronological split:
- the scaler is fitted on the training rows only and saved with the weights;
- the last 15% of the training rows are a validation slice, used to keep the
  best epoch and to measure the residual spread that sets the interval width;
- nothing after `train_end` is ever seen during fitting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils.data_pipeline import enrich_dataset


class StackedLSTM(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def features(close: np.ndarray) -> np.ndarray:
    """Causal features: every row uses prices up to and including that row only."""
    df = enrich_dataset(pd.DataFrame({"close": close}))
    log_ret = np.concatenate([[0.0], np.diff(np.log(close))])
    return np.column_stack([log_ret, df["rsi"].to_numpy() / 100.0, df["volatility"].to_numpy()])


@dataclass
class LSTMForecaster:
    seq_len: int = 30
    hidden_dim: int = 32
    epochs: int = 60
    lr: float = 3e-3
    seed: int = 0
    val_frac: float = 0.15
    model: StackedLSTM | None = field(default=None, repr=False)

    def _windows(self, f: np.ndarray, targets: range):
        idx = [j for j in targets if j - self.seq_len >= 1]
        x = np.stack([f[j - self.seq_len : j] for j in idx])
        y = f[idx, 0]  # next log return
        return idx, x, y

    def fit(self, close: np.ndarray, train_end: int | None = None) -> "LSTMForecaster":
        torch.manual_seed(self.seed)
        train_end = train_end or len(close)
        f = features(close[:train_end])
        val_start = int(train_end * (1 - self.val_frac))
        self.mean_ = f[:val_start].mean(0)
        self.std_ = f[:val_start].std(0) + 1e-8
        fs = (f - self.mean_) / self.std_
        _, xtr, ytr = self._windows(fs, range(1, val_start))
        _, xva, yva = self._windows(fs, range(val_start, train_end))
        xtr, ytr = torch.tensor(xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
        xva, yva = torch.tensor(xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)

        self.model = StackedLSTM(f.shape[1], self.hidden_dim)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        best, best_state = np.inf, None
        for _ in range(self.epochs):
            self.model.train()
            opt.zero_grad()
            loss = loss_fn(self.model(xtr).squeeze(1), ytr)
            loss.backward()
            opt.step()
            self.model.eval()
            with torch.no_grad():
                val = loss_fn(self.model(xva).squeeze(1), yva).item()
            if val < best:
                best, best_state = val, {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(best_state)
        self.model.eval()
        with torch.no_grad():
            resid = (self.model(xva).squeeze(1) - yva).numpy() * self.std_[0]
        # Residual spread of the predicted log return on validation data: this, not a constant, sets the bands.
        self.resid_std_ = float(np.std(resid))
        self.train_end_ = train_end
        return self

    def predict_return(self, close: np.ndarray) -> float:
        """Predicted log return for the step after the last price in `close`."""
        fs = (features(close) - self.mean_) / self.std_
        x = torch.tensor(fs[-self.seq_len :][None], dtype=torch.float32)
        with torch.no_grad():
            return float(self.model(x).item() * self.std_[0] + self.mean_[0])

    def one_step_predictions(self, close: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Price forecast for each index j, using prices strictly before j."""
        return np.array([close[j - 1] * np.exp(self.predict_return(close[:j])) for j in idx])

    def forecast(self, close: np.ndarray, horizon: int):
        """Recursive multi-step forecast. Interval: ±1.96 × validation residual std × √h, in log space."""
        path = list(close)
        out = []
        for h in range(1, horizon + 1):
            nxt = path[-1] * np.exp(self.predict_return(np.asarray(path)))
            path.append(nxt)
            half = 1.96 * self.resid_std_ * np.sqrt(h)
            out.append({"price": float(nxt), "lower": float(nxt * np.exp(-half)), "upper": float(nxt * np.exp(half))})
        return out

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict(), "mean": self.mean_, "std": self.std_,
                    "resid_std": self.resid_std_, "seq_len": self.seq_len, "hidden_dim": self.hidden_dim}, path)

    @classmethod
    def load(cls, path: str) -> "LSTMForecaster":
        ck = torch.load(path, weights_only=False)
        m = cls(seq_len=ck["seq_len"], hidden_dim=ck["hidden_dim"])
        m.model = StackedLSTM(3, ck["hidden_dim"])
        m.model.load_state_dict(ck["state_dict"])
        m.model.eval()
        m.mean_, m.std_, m.resid_std_ = ck["mean"], ck["std"], ck["resid_std"]
        return m

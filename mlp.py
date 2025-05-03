import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy
from typing import Optional, Literal

from tqdm import trange
from numpy.typing import NDArray

from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from sklearn.utils import check_array, check_X_y
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin

from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins


def _train_clf(
    model: nn.Module,
    device: Literal["cpu", "cuda", "mps"],
    dataloaders: dict[Literal["train", "valid"], DataLoader],
    lr: float,
    max_iter: int,
    patience: Optional[int] = None,
    verbose: bool = False,
) -> dict[Literal["train", "valid"], list[float]]:

    model.to(device)

    # --- Training loop
    criterion = F.cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    best_model_params = None
    steps_without_improvement = 0

    acc_history = {"train": [], "valid": []}

    for iter in (pbar := trange(max_iter, disable=not verbose)):
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            if phase == "valid":
                model.eval()

            running_hits = 0
            running_total = 0

            for X_batch, y_batch in dataloaders[phase]:
                X_batch: Tensor = X_batch.to(device)
                y_batch: Tensor = y_batch.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    # Forward
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    # Backward
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_hits += (torch.argmax(logits, dim=1) == y_batch).sum().item()
                running_total += y_batch.size(0)

            # Compute epoch statistics
            acc = running_hits / running_total
            acc_history[phase].append(acc)

            # Update best model
            if phase == "valid" and acc > best_acc:
                steps_without_improvement = 0
                best_acc, best_model_params = acc, deepcopy(model.state_dict())
            elif phase == "valid":
                steps_without_improvement += 1

        # Log progress
        desc = (
            f"Iter {iter+1}/{max_iter}"
            f" | Train acc. {acc_history['train'][-1]:.2%}"
            f" | Valid acc. {acc_history['valid'][-1]:.2%}"
            f" | Best acc. {best_acc:.2%}"
        )
        pbar.set_description(desc)

        # Early stopping
        if patience and steps_without_improvement > patience:
            pbar.set_description(desc + " | Early stopping!")
            break

    # Load best model params
    model.load_state_dict(best_model_params)

    return acc_history


class BinnedMLPClassifier(nn.Module, ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        dim_emb: int,
        dim_hid: int,
        num_hid: int,
        num_bins: int,
        dropout: float,
        lr: float,
        max_iter: int,
        batch_size: int,
        valid_frac: float,
        patience: Optional[int] = None,
        verbose: bool = False,
    ):
        super().__init__()

        # --- Perform basic sanity checks
        assert isinstance(dim_emb, int) and dim_hid > 0
        assert isinstance(dim_hid, int) and dim_hid > 0
        assert isinstance(num_hid, int) and num_hid >= 0
        assert isinstance(dropout, float) and 0 <= dropout < 1
        assert isinstance(lr, float) and lr > 0
        assert isinstance(max_iter, int) and max_iter > 0
        assert isinstance(batch_size, int) and batch_size > 0
        assert isinstance(valid_frac, float) and 0 < valid_frac < 1
        assert patience is None or (isinstance(patience, int) and patience > 0)

        # Network architecture
        self.dim_hid: int = dim_hid
        self.num_hid: int = num_hid

        # Piecewise Linear Embedding params
        self.dim_emb: int = dim_emb
        self.num_bins: int = num_bins

        # Training hyperparams
        self.lr: float = lr
        self.dropout: float = dropout
        self.max_iter: int = max_iter
        self.batch_size: int = batch_size
        self.valid_frac: float = valid_frac
        self.patience: Optional[int] = patience

        # Other params
        self.verbose: bool = verbose

    def _build(self):
        self.device: Literal["cuda", "mps", "cpu"]

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.embedding = PiecewiseLinearEmbeddings(self.bins, self.dim_emb, activation=False, version="B")

        self.layers = []
        self.layers.append(nn.Linear(self.dim_emb * self.dim_in, self.dim_hid))
        self.layers.append(nn.BatchNorm1d(self.dim_hid))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=self.dropout))

        for _ in range(self.num_hid):
            self.layers.append(nn.Linear(self.dim_hid, self.dim_hid))
            self.layers.append(nn.BatchNorm1d(self.dim_hid))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=self.dropout))

        self.layers.append(nn.Linear(self.dim_hid, self.dim_out))
        self.layers = nn.Sequential(*self.layers)

        self.to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x).flatten(1)
        return self.layers(x)

    def fit(self, X: NDArray, y: NDArray):
        X = np.array(X)
        y = np.array(y)
        check_X_y(X, y)

        self.dim_in = X.shape[1]
        self.dim_out = len(np.unique(y))

        # --- Get validation data
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=self.valid_frac,
            shuffle=True,
            stratify=y,
        )

        # --- Map labels to int
        self.label_encoder_ = LabelEncoder().fit(y)
        y_train = self.label_encoder_.transform(y_train)
        y_valid = self.label_encoder_.transform(y_valid)

        # --- To Tensor
        X_train = torch.tensor(X_train).float()
        X_valid = torch.tensor(X_valid).float()
        y_train = torch.tensor(y_train)
        y_valid = torch.tensor(y_valid)

        # --- Build network
        self.bins = compute_bins(X=X_train, n_bins=self.num_bins)
        self._build()

        # --- Create dataloaders
        dataloaders = {
            "train": DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True),
            "valid": DataLoader(TensorDataset(X_valid, y_valid), batch_size=self.batch_size, shuffle=True),
        }
        # --- Train clf
        self.acc_history_ = _train_clf(
            model=self,
            device=self.device,
            dataloaders=dataloaders,
            lr=self.lr,
            max_iter=self.max_iter,
            patience=self.patience,
            verbose=self.verbose,
        )

        return self

    @torch.no_grad()
    def predict(self, X: NDArray) -> NDArray:
        # --- Perform basic sanity checks
        X = np.array(X)
        check_array(X)

        # --- Predict
        self.eval()

        logits = self(torch.tensor(X).float().to(self.device))
        logits = logits.to("cpu").numpy()
        labels = np.argmax(logits, axis=1)

        return self.label_encoder_.inverse_transform(labels)


class RawMLPClassifier(nn.Module, ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        dim_hid: int,
        num_hid: int,
        dropout: float,
        lr: float,
        max_iter: int,
        batch_size: int,
        valid_frac: float,
        patience: Optional[int] = None,
        verbose: bool = False,
    ):
        super().__init__()

        # --- Perform basic sanity checks
        assert isinstance(dim_hid, int) and dim_hid > 0
        assert isinstance(num_hid, int) and num_hid >= 0
        assert isinstance(dropout, float) and 0 <= dropout < 1
        assert isinstance(lr, float) and lr > 0
        assert isinstance(max_iter, int) and max_iter > 0
        assert isinstance(batch_size, int) and batch_size > 0
        assert isinstance(valid_frac, float) and 0 < valid_frac < 1
        assert patience is None or (isinstance(patience, int) and patience > 0)

        # Network architecture
        self.dim_hid: int = dim_hid
        self.num_hid: int = num_hid

        # Training hyperparams
        self.lr: float = lr
        self.dropout: float = dropout
        self.max_iter: int = max_iter
        self.batch_size: int = batch_size
        self.valid_frac: float = valid_frac
        self.patience: Optional[int] = patience

        # Other params
        self.verbose: bool = verbose

    def _build(self):
        self.device: Literal["cuda", "mps", "cpu"]
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.layers = []
        self.layers.append(nn.Linear(self.dim_in, self.dim_hid))
        self.layers.append(nn.BatchNorm1d(self.dim_hid))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=self.dropout))

        for _ in range(self.num_hid):
            self.layers.append(nn.Linear(self.dim_hid, self.dim_hid))
            self.layers.append(nn.BatchNorm1d(self.dim_hid))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=self.dropout))

        self.layers.append(nn.Linear(self.dim_hid, self.dim_out))
        self.layers = nn.Sequential(*self.layers)

        self.to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    def fit(self, X: NDArray, y: NDArray):
        X = np.array(X)
        y = np.array(y)
        check_X_y(X, y)

        # --- Set additional fields
        self.dim_in: int = X.shape[1]
        self.dim_out: int = len(np.unique(y))
        self.acc_history_: Optional[dict[str, list[float]]] = None
        self.label_encoder_: Optional[LabelEncoder] = None

        # --- Get validation data
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=self.valid_frac,
            shuffle=True,
            stratify=y,
        )

        # --- Map labels to int
        self.label_encoder_ = LabelEncoder().fit(y)
        y_train = self.label_encoder_.transform(y_train)
        y_valid = self.label_encoder_.transform(y_valid)

        # --- To Tensor
        X_train = torch.tensor(X_train).float()
        X_valid = torch.tensor(X_valid).float()
        y_train = torch.tensor(y_train)
        y_valid = torch.tensor(y_valid)

        # --- Build network
        self._build()

        # --- Create dataloaders
        dataloaders = {
            "train": DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True),
            "valid": DataLoader(TensorDataset(X_valid, y_valid), batch_size=self.batch_size, shuffle=True),
        }
        # --- Train clf
        self.acc_history_ = _train_clf(
            model=self,
            device=self.device,
            dataloaders=dataloaders,
            lr=self.lr,
            max_iter=self.max_iter,
            patience=self.patience,
            verbose=self.verbose,
        )

        return self

    @torch.no_grad()
    def predict(self, X: NDArray) -> NDArray:
        # --- Perform basic sanity checks
        X = np.array(X)
        check_array(X)

        # --- Predict
        self.eval()

        logits = self(torch.tensor(X).float().to(self.device))
        logits = logits.to("cpu").numpy()
        labels = np.argmax(logits, axis=1)

        return self.label_encoder_.inverse_transform(labels)


if __name__ == "__main__":
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(
        BinnedMLPClassifier(
            dim_emb=8,
            dim_hid=256,
            num_hid=2,
            num_bins=8,
            dropout=0.1,
            lr=1e-3,
            max_iter=200,
            batch_size=256,
            valid_frac=0.2,
            patience=50,
        )
    )

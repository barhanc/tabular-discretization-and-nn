import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy
from typing import Optional
from itertools import pairwise
from typing_extensions import Self

from tqdm import trange
from torch import Tensor
from numpy.typing import NDArray
from sklearn.utils import check_array, check_X_y
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin


def make_batch(X: NDArray, y: NDArray, batch_size: int):
    samples = np.random.choice(X.shape[0], (X.shape[0] // batch_size, batch_size), replace=False)
    for sample in samples:
        yield torch.tensor(X[sample, ...]).float(), torch.tensor(y[sample, ...])


class MLPClassifier(nn.Module, BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layer_size: int,
        num_hidden_layers: int,
        dropout_rate: float,
        learning_rate: float,
        max_iter: int,
        batch_size: int,
        validation_fraction: float,
        early_stopping: bool = False,
        patience: Optional[int] = None,
        random_state: Optional[int] = None,
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ):
        super().__init__()

        # --- Perform basic sanity checks
        assert isinstance(in_features, int) and in_features > 0
        assert isinstance(out_features, int) and out_features > 0
        assert isinstance(hidden_layer_size, int) and hidden_layer_size > 0
        assert isinstance(num_hidden_layers, int) and num_hidden_layers >= 0
        assert isinstance(dropout_rate, float) and 0 <= dropout_rate < 1
        assert isinstance(learning_rate, float) and learning_rate > 0
        assert isinstance(max_iter, int) and max_iter > 0
        assert isinstance(batch_size, int) and batch_size > 0
        assert isinstance(validation_fraction, float) and 0 < validation_fraction < 1
        assert patience is None or (isinstance(patience, int) and patience > 0)
        assert random_state is None or isinstance(random_state, int)
        assert device is None or isinstance(device, torch.device)

        # Network architecture
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.hidden_layer_size: int = hidden_layer_size
        self.num_hidden_layers: int = num_hidden_layers

        # Training hyperparams
        self.dropout_rate: float = dropout_rate
        self.learning_rate: float = learning_rate
        self.max_iter: int = max_iter
        self.batch_size: int = batch_size
        self.validation_fraction: float = validation_fraction
        self.early_stopping: bool = early_stopping
        self.patience: Optional[int] = patience
        self.random_state: Optional[int] = random_state
        self.verbose: bool = verbose

        self.device: torch.device

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif torch.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.acc_hist_: Optional[dict[str, list[float]]] = None
        self.label_encoder_: Optional[LabelEncoder] = None

        # --- Build feed-forward network
        hidden_layers_sizes = [self.in_features]
        hidden_layers_sizes += [self.hidden_layer_size for _ in range(self.num_hidden_layers)]
        hidden_layers_sizes += [self.out_features]

        layers = []
        for i, (n_in, n_out) in enumerate(pairwise(hidden_layers_sizes)):
            layers.append(nn.Linear(in_features=n_in, out_features=n_out))
            if i < len(hidden_layers_sizes) - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=self.dropout_rate))

        self._layers = nn.Sequential(*layers)

        # --- Model to `self.device`
        self.to(self.device)

        # --- Set random seed
        if self.random_state:
            torch.manual_seed(seed=self.random_state)
            np.random.seed(seed=self.random_state)
            random.seed(self.random_state)

    def forward(self, X: Tensor) -> Tensor:
        return self._layers(X)

    def fit(self, X: NDArray, y: NDArray) -> Self:
        X = np.array(X)
        y = np.array(y)
        check_X_y(X, y)

        # --- Get validation data
        X_train, X_valid, t_train, t_valid = train_test_split(
            X,
            y,
            test_size=self.validation_fraction,
            shuffle=True,
            stratify=y,
            random_state=self.random_state,
        )

        # --- Map labels to int
        self.label_encoder_ = LabelEncoder().fit(y)
        t_train = self.label_encoder_.transform(t_train)
        t_valid = self.label_encoder_.transform(t_valid)

        # --- Training loop
        criterion = F.cross_entropy
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        best_acc = 0.0
        best_model_params = None
        steps_without_improvement = 0

        self.acc_hist_ = {"acc_train": [], "acc_valid": []}

        for iter in (pbar := trange(self.max_iter, disable=not self.verbose)):
            # --- Training phase
            self.train()
            running_hits = 0
            running_total = 0

            for X_batch, t_batch in make_batch(X_train, t_train, batch_size=self.batch_size):
                optimizer.zero_grad()

                # Forward
                X_batch = X_batch.to(self.device)
                t_batch = t_batch.to(self.device)
                y_batch = self(X_batch)

                # Backward
                loss = criterion(y_batch, t_batch)
                loss.backward()

                optimizer.step()

                # Update running statistics
                running_hits += (torch.argmax(y_batch, dim=1) == t_batch).sum().item()
                running_total += t_batch.size(0)

            acc_train = running_hits / running_total
            self.acc_hist_["acc_train"].append(acc_train)

            # --- Validation phase
            self.eval()

            with torch.no_grad():
                logits = self(torch.tensor(X_valid).float().to(self.device))
                logits = logits.to("cpu").numpy()

            acc_valid = np.sum(np.argmax(logits, axis=1) == t_valid) / t_valid.size
            self.acc_hist_["acc_valid"].append(acc_valid)

            if best_acc < acc_valid:
                best_acc = acc_valid
                best_model_params = deepcopy(self.state_dict())
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            # --- Log progress
            desc = f"Iter {iter+1}/{self.max_iter}"
            desc += f" | Train acc. {acc_train:.2%}"
            desc += f" | Valid acc. {acc_valid:.2%}"
            desc += f" | Best  acc. {best_acc: .2%}"
            pbar.set_description(desc)

            # --- Early stopping
            if self.early_stopping and steps_without_improvement > self.patience:
                pbar.set_description(desc + " | Early stopping!")
                break

        self.load_state_dict(best_model_params)

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

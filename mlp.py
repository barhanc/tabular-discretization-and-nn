import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from warnings import warn
from numbers import Integral, Real
from typing import Optional, Literal

from numpy.typing import NDArray
from numpy.random import RandomState

from tqdm import trange
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_random_state, check_is_fitted, validate_data

__all__ = [
    "MLPClassifier",
    "QPLEMLPClassifier",
]


class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_hid: int,
        num_hid: int,
        dropout: float,
    ):
        super().__init__()

        self.layers = []
        self.layers.append(nn.Linear(dim_in, dim_hid))
        self.layers.append(nn.BatchNorm1d(dim_hid))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=dropout))

        for _ in range(num_hid):
            self.layers.append(nn.Linear(dim_hid, dim_hid))
            self.layers.append(nn.BatchNorm1d(dim_hid))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout))

        self.layers.append(nn.Linear(dim_hid, dim_out))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


def _train_model(
    model: nn.Module,
    device: Literal["cpu", "cuda", "mps"],
    dataloaders: dict[Literal["train", "valid"], DataLoader],
    lr: float,
    max_iter: int,
    patience: int,
    regression: bool = False,
    verbose: bool = False,
) -> tuple[dict[Literal["train", "valid"], list[float]], int]:
    # Training loop
    model.to(device)

    criterion = nn.MSELoss() if regression else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

    return acc_history, iter + 1


class MLPClassifier(ClassifierMixin, BaseEstimator):
    _parameter_constraints = {
        "dim_hid": [Interval(Integral, 1, None, closed="left")],
        "num_hid": [Interval(Integral, 1, None, closed="left")],
        "dropout": [Interval(Real, 0, 1, closed="left")],
        "lr": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "valid_frac": [Interval(Real, 0, 1, closed="left")],
        "patience": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "verbose": ["boolean"],
        "device": [StrOptions({"cpu", "cuda", "mps", "auto"})],
    }

    def __init__(
        self,
        dim_hid: int = 128,
        num_hid: int = 2,
        dropout: float = 0.2,
        lr: float = 3e-4,
        max_iter: int = 200,
        batch_size: int = 256,
        valid_frac: float = 0.2,
        patience: int = 40,
        random_state: Optional[int | RandomState] = None,
        verbose: bool = False,
        device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
    ):
        super().__init__()

        self.dim_hid: int = dim_hid
        self.num_hid: int = num_hid

        self.lr: float = lr
        self.dropout: float = dropout
        self.max_iter: int = max_iter
        self.batch_size: int = batch_size
        self.valid_frac: float = valid_frac
        self.patience: int = patience

        self.random_state: Optional[int | RandomState] = random_state
        self.device: Literal["cpu", "cuda", "mps", "auto"] = device
        self.verbose: bool = verbose

    def fit(self, X, y):
        X, y = validate_data(self, X, y, accept_sparse=False, reset=True)
        check_classification_targets(y)
        X, y = np.array(X), np.array(y)

        self.label_encoder_: LabelEncoder = LabelEncoder().fit(y)
        self.classes_: NDArray = self.label_encoder_.classes_
        self.n_classes_: int = len(self.classes_)

        self._random_state: RandomState = check_random_state(self.random_state)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=self.valid_frac,
            random_state=self._random_state,
            shuffle=True,
            stratify=y,
        )

        y_train = self.label_encoder_.transform(y_train)
        y_valid = self.label_encoder_.transform(y_valid)

        self.device_: Literal["cuda", "mps", "cpu"]

        if self.device == "auto":
            if torch.cuda.is_available():
                self.device_ = "cuda"
            elif torch.mps.is_available():
                self.device_ = "mps"
            else:
                self.device_ = "cpu"
        else:
            if self.device == "cuda" and not torch.cuda.is_available():
                raise ValueError(f"{self.device=} passed but CUDA not available!")
            if self.device == "mps" and not torch.mps.is_available():
                raise ValueError(f"{self.device=} passed but MPS not available!")
            self.device_ = self.device

        torch.manual_seed(self._random_state.randint(2**32))

        X_train = torch.tensor(X_train).float()
        X_valid = torch.tensor(X_valid).float()
        y_train = torch.tensor(y_train)
        y_valid = torch.tensor(y_valid)

        dataloaders = {
            "train": DataLoader(TensorDataset(X_train, y_train), self.batch_size, shuffle=True),
            "valid": DataLoader(TensorDataset(X_valid, y_valid), self.batch_size, shuffle=True),
        }

        self._model: nn.Module = MLP(
            dim_in=self.n_features_in_,
            dim_out=self.n_classes_,
            dim_hid=self.dim_hid,
            num_hid=self.num_hid,
            dropout=self.dropout,
        )

        self.acc_history_, self.n_iter_ = _train_model(
            model=self._model,
            device=self.device_,
            dataloaders=dataloaders,
            lr=self.lr,
            max_iter=self.max_iter,
            patience=self.patience,
            regression=False,
            verbose=self.verbose,
        )

        return self

    def predict(self, X: NDArray) -> NDArray:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, accept_sparse=False, accept_large_sparse=False)
        X = np.array(X)

        self._model.to(self.device_)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(torch.tensor(X).float().to(self.device_))
            logits = logits.to("cpu").numpy()
            labels = np.argmax(logits, axis=1)
        return self.label_encoder_.inverse_transform(labels)


class QPLEMLPClassifier(ClassifierMixin, BaseEstimator):
    _parameter_constraints = {
        "dim_hid": [Interval(Integral, 1, None, closed="left")],
        "dim_emb": [Interval(Integral, 1, None, closed="left")],
        "num_hid": [Interval(Integral, 1, None, closed="left")],
        "num_bins": [Interval(Integral, 1, None, closed="neither")],
        "dropout": [Interval(Real, 0, 1, closed="left")],
        "lr": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "valid_frac": [Interval(Real, 0, 1, closed="left")],
        "patience": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "verbose": ["boolean"],
        "device": [StrOptions({"cpu", "cuda", "mps", "auto"})],
    }

    def __init__(
        self,
        dim_hid: int = 128,
        dim_emb: int = 8,
        num_hid: int = 2,
        num_bins: int = 64,
        dropout: float = 0.2,
        lr: float = 3e-4,
        max_iter: int = 200,
        batch_size: int = 256,
        valid_frac: float = 0.2,
        patience: int = 40,
        random_state: Optional[int | RandomState] = None,
        verbose: bool = False,
        device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
    ):
        super().__init__()

        self.dim_hid: int = dim_hid
        self.dim_emb: int = dim_emb
        self.num_hid: int = num_hid
        self.num_bins: int = num_bins

        self.lr: float = lr
        self.dropout: float = dropout
        self.max_iter: int = max_iter
        self.batch_size: int = batch_size
        self.valid_frac: float = valid_frac
        self.patience: int = patience

        self.random_state: Optional[int | RandomState] = random_state
        self.device: Literal["cpu", "cuda", "mps", "auto"] = device
        self.verbose: bool = verbose

    def fit(self, X, y):
        X, y = validate_data(self, X, y, accept_sparse=False, reset=True)
        check_classification_targets(y)
        X, y = np.array(X), np.array(y)

        self.label_encoder_: LabelEncoder = LabelEncoder().fit(y)
        self.classes_: NDArray = self.label_encoder_.classes_
        self.n_classes_: int = len(self.classes_)

        self._random_state: RandomState = check_random_state(self.random_state)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=self.valid_frac,
            random_state=self._random_state,
            shuffle=True,
            stratify=y,
        )

        y_train = self.label_encoder_.transform(y_train)
        y_valid = self.label_encoder_.transform(y_valid)

        self.device_: Literal["cuda", "mps", "cpu"]

        if self.device == "auto":
            if torch.cuda.is_available():
                self.device_ = "cuda"
            elif torch.mps.is_available():
                self.device_ = "mps"
            else:
                self.device_ = "cpu"
        else:
            if self.device == "cuda" and not torch.cuda.is_available():
                raise ValueError(f"{self.device=} passed but CUDA not available!")
            if self.device == "mps" and not torch.mps.is_available():
                raise ValueError(f"{self.device=} passed but MPS not available!")
            self.device_ = self.device

        torch.manual_seed(self._random_state.randint(2**32))

        X_train = torch.tensor(X_train).float()
        X_valid = torch.tensor(X_valid).float()
        y_train = torch.tensor(y_train)
        y_valid = torch.tensor(y_valid)

        dataloaders = {
            "train": DataLoader(TensorDataset(X_train, y_train), self.batch_size, shuffle=True),
            "valid": DataLoader(TensorDataset(X_valid, y_valid), self.batch_size, shuffle=True),
        }

        if len(X_train) - 1 < self.num_bins:
            self._num_bins = len(X_train) - 1
            warn(
                f"num_bins must be more than 1, but less than len(X_train), "
                f"however: {self.num_bins=}, {len(X_train)=}. "
                f"Setting num_bins to len(X_train) - 1."
            )
        else:
            self._num_bins = len(X_train) - 1

        self._model: nn.Module = nn.Sequential(
            # We use the default version recommended in the README
            # https://github.com/yandex-research/rtdl-num-embeddings/tree/main/package#which-embeddings-to-choose
            PiecewiseLinearEmbeddings(
                bins=compute_bins(X_train, n_bins=self._num_bins),
                d_embedding=self.dim_emb,
                activation=False,
                version="B",
            ),
            nn.Flatten(1),
            MLP(
                dim_in=self.dim_emb * self.n_features_in_,
                dim_out=self.n_classes_,
                dim_hid=self.dim_hid,
                num_hid=self.num_hid,
                dropout=self.dropout,
            ),
        )

        self.acc_history_, self.n_iter_ = _train_model(
            model=self._model,
            device=self.device_,
            dataloaders=dataloaders,
            lr=self.lr,
            max_iter=self.max_iter,
            patience=self.patience,
            regression=False,
            verbose=self.verbose,
        )

        return self

    def predict(self, X: NDArray) -> NDArray:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, accept_sparse=False, accept_large_sparse=False)
        X = np.array(X)

        self._model.to(self.device_)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(torch.tensor(X).float().to(self.device_))
            logits = logits.to("cpu").numpy()
            labels = np.argmax(logits, axis=1)
        return self.label_encoder_.inverse_transform(labels)


if __name__ == "__main__":
    from warnings import simplefilter
    from sklearn.utils.estimator_checks import check_estimator

    simplefilter("ignore", UserWarning)
    check_estimator(MLPClassifier())
    check_estimator(QPLEMLPClassifier())

import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from numbers import Integral, Real
from typing import Optional, Literal

from numpy.typing import NDArray
from numpy.random import RandomState

from tqdm import trange
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from rtdl_num_embeddings import compute_bins, PiecewiseLinearEncoding

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import Interval, StrOptions, validate_parameter_constraints
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_random_state, check_is_fitted, validate_data


class _TorchMLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dim_hid: int, n_hid: int, dropout: float):
        super().__init__()

        self.layers = []
        self.layers.append(nn.Linear(dim_in, dim_hid))
        self.layers.append(nn.BatchNorm1d(dim_hid))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=dropout))

        for _ in range(n_hid):
            self.layers.append(nn.Linear(dim_hid, dim_hid))
            self.layers.append(nn.BatchNorm1d(dim_hid))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout))

        self.layers.append(nn.Linear(dim_hid, dim_out))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


def _train_clf(
    model: nn.Module,
    device: Literal["cpu", "cuda", "mps"],
    dataloaders: dict[Literal["train", "valid"], DataLoader],
    learning_rate: float,
    weight_decay: float,
    max_iter: int,
    tol: float,
    n_iter_no_change: int,
    early_stopping: bool,
    verbose: bool = False,
) -> tuple[dict[Literal["train", "valid"], list[float]], int]:

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_acc = 0.0
    best_model_params = deepcopy(model.state_dict())
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

            if phase == "valid" and acc > best_acc:
                steps_without_improvement = 0 if acc - best_acc >= tol else steps_without_improvement + 1
                best_acc, best_model_params = acc, deepcopy(model.state_dict())
            elif phase == "valid":
                steps_without_improvement += 1

        # Log progress
        desc = (
            f"Iter {iter+1}/{max_iter} "
            f"| Train acc. {acc_history['train'][-1]:.2%} "
            f"| Valid acc. {acc_history['valid'][-1]:.2%} "
            f"| Best acc. {best_acc:.2%} "
        )
        pbar.set_description(desc)

        # Early stopping
        if early_stopping and steps_without_improvement > n_iter_no_change:
            pbar.set_description(desc + " | Early stopping!")
            break

    # Load best model params
    model.load_state_dict(best_model_params)

    return acc_history, iter + 1


class MLPClassifier(ClassifierMixin, BaseEstimator):
    """
    Multilayer Perceptron classifier with optional quantile-based Piecewise Linear Encoding (PLE).

    This class implements the MLP classifier with optional quantile-based Piecewise Linear Encoding
    layer proposed in [1]. To see the exact architecture of the used MLP scroll down to Diagram
    section.

    Parameters
    ----------
    dim_hid : int,
        Size of each hidden layer of the MLP. For simplicity we assume that every hidden layer has
        the same size.

    n_hid : int,
        Number of hidden layers. The total number of layers in the MLP is `n_hid + 2`.

    max_bins : int,
        Max number of bins to use when discretizing the features based on quantiles for the
        quantile-based Piecewise Linear Encoding. Only effective when `use_quantile_encoding=True`.

    dropout : float,
        Dropout probability.

    lr : float,
        Learning rate.

    weight_decay : float,
        Strength of the L2 regularization term used in the AdamW optimizer.

    max_iter : int,
        Maximum number of iterations. The optimizer iterates until convergence (determined by `tol`)
        or this number of iterations

    batch_size : int, "auto",
        Size of the minibatches. When set to "auto" then `batch_size=min(n_samples, 256)` if
        `n_samples < 50_000` and `1024` otherwise.

    validation_fraction : float,
        The proportion of training data to set aside as validation set for early stopping. Must be
        between 0 and 1.

    tol : float,
        Tolerance for the optimization. When the loss is not improving by at least `tol` for
        `n_iter_no_change` consecutive iterations, convergence is considered to be reached and
        training stops.

    n_iter_no_change : int,
        Maximum number of epochs to not meet tol improvement.

    early_stopping : bool,
        Whether to use early stopping to terminate training when validation score is not improving.
        If set to true, it will terminate training when validation score is not improving by at
        least `tol` for `n_iter_no_change` consecutive epochs

    use_quantile_encoding : bool,
        Whether to use quantile-based Piecewise Linear Encoding layer described in [1]. If set to
        `True` it will first compute the appropriate bins on the training set (after setting aside
        the validation set) and then add encoding layer to the model.

    random_state : int, RandomState instance,
        Determines random number generation for weights and bias initialization, train-test split,
        and batch sampling. Pass an int for reproducible results across multiple function calls.

    verbose : bool,
        Whether to print progress messages to stdout.

    device : {"cpu", "cuda", "mps", "auto"},
        `torch.device` used for inference and training. If "auto" then the device is chosen
        automatically based on the availability in the following order "cuda", "mps", "cpu".

    Attributes
    ----------
    classes_ : ndarray or list of ndarray of shape `(n_classes,)`
        Class labels for each output.

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape `(n_features_in_,)`
        Names of features seen during fit. Defined only when X has feature names that are all
        strings.

    n_iter_ : int
        The number of iterations the optimizer has run.

    batch_size_ : int
        Final batch size used for inference and training.

    acc_history_ : dict
        Dictionary with keys `"train"` and `"valid"` which contains the history of accuracy scores
        on both training and validation sets obtained during training.

    device_ : {"cpu", "cuda", "mps"}
        Final device used for inference and training.

    Diagram
    -------
    ```
    [Input(n_samples, n_features)]
        |
    ***** optional
    | [qPLE(n_bins)]
    *****
        |
    [Linear(dim_hid)]
        |
    [Batchnorm(dim_hid)]
        |
    [ReLU]
        |
    [Dropout]
        |
    ***** n_hid x
    | [Linear(dim_hid)]
    |     |
    | [Batchnorm(dim_hid)]
    |     |
    | [ReLU]
    |     |
    | [Dropout]
    *****
        |
    [Linear(n_classes)]
        |
    [Output(n_samples, n_classes)]
    ```

    Notes
    -----
    This class uses PyTorch library to implement and train the network itself. Additionally, for the
    quantile-based Piecewise Linear Encoding layer we use the code published by the authors of the
    paper [1] which is available here: https://github.com/yandex-research/rtdl-num-embeddings.

    TODO: Write about sane defaults...

    References
    ----------
    [1] Yury Gorishniy, Ivan Rubachev, & Artem Babenko. (2023). On Embeddings for Numerical Features
        in Tabular Deep Learning. https://arxiv.org/pdf/2203.05556

    """

    _parameter_constraints = {
        "dim_hid": [Interval(Integral, 1, None, closed="left")],
        "n_hid": [Interval(Integral, 1, None, closed="left")],
        "n_bins": [Interval(Integral, 1, None, closed="neither")],
        "dropout": [Interval(Real, 0, 1, closed="left")],
        "learning_rate": [Interval(Real, 0, None, closed="left")],
        "weight_decay": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "batch_size": [StrOptions({"auto"}), Interval(Integral, 1, None, closed="left")],
        "validation_fraction": [Interval(Real, 0, 1, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "n_iter_no_change": [Interval(Integral, 1, None, closed="left")],
        "use_quantile_encoding": ["boolean"],
        "random_state": ["random_state"],
        "verbose": ["boolean"],
        "early_stopping": ["boolean"],
        "device": [StrOptions({"cpu", "cuda", "mps", "auto"})],
    }

    def __init__(
        self,
        dim_hid: int = 256,
        n_hid: int = 1,
        max_bins: int = 48,
        dropout: float = 0.5,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        max_iter: int = 200,
        batch_size: int | Literal["auto"] = "auto",
        validation_fraction: float = 0.1,
        tol: float = 1e-4,
        n_iter_no_change: int = 40,
        early_stopping: bool = True,
        use_quantile_encoding: bool = True,
        random_state: Optional[int | RandomState] = None,
        verbose: bool = False,
        device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
    ):
        super().__init__()

        self.dim_hid: int = dim_hid
        self.n_hid: int = n_hid
        self.max_bins: int = max_bins

        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_decay
        self.dropout: float = dropout
        self.max_iter: int = max_iter
        self.batch_size: int = batch_size
        self.validation_fraction: float = validation_fraction
        self.tol: float = tol
        self.n_iter_no_change: int = n_iter_no_change
        self.early_stopping: bool = early_stopping
        self.use_quantile_encoding: bool = use_quantile_encoding

        self.random_state: Optional[int | RandomState] = random_state
        self.device: Literal["cpu", "cuda", "mps", "auto"] = device
        self.verbose: bool = verbose

    def _set_device(self):
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

    def fit(self, X, y):
        # --- Perform sanity checks and make sure we are using ndarrays
        validate_parameter_constraints(
            self._parameter_constraints,
            self.get_params(),
            self.__class__.__name__,
        )
        X, y = validate_data(self, X, y, accept_sparse=False, reset=True)
        check_classification_targets(y)
        X, y = np.array(X), np.array(y)

        # --- Turn seed into a RandomState instance
        self._random_state: RandomState = check_random_state(self.random_state)

        # --- Split data into train and validation sets for monitoring and early stopping
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=self.validation_fraction,
            random_state=self._random_state,
            shuffle=True,
            stratify=y,
        )

        # --- Encode labels
        self._label_encoder = LabelEncoder().fit(y)
        self.classes_ = self._label_encoder.classes_
        y_train = self._label_encoder.transform(y_train)
        y_valid = self._label_encoder.transform(y_valid)

        # --- Prepare data and model for training using PyTorch
        self._set_device()
        torch.manual_seed(self._random_state.randint(2**32))

        X_train = torch.tensor(X_train).float()
        X_valid = torch.tensor(X_valid).float()
        y_train = torch.tensor(y_train)
        y_valid = torch.tensor(y_valid)

        if self.batch_size == "auto":
            if len(X) >= 50_000:
                self.batch_size_ = 1024
            else:
                self.batch_size_ = min(len(X), 256)
        else:
            if self.batch_size > len(X):
                raise ValueError(f"{self.batch_size=} is greater than {len(X)=}")
            self.batch_size_ = self.batch_size

        dataloaders = {
            "train": DataLoader(TensorDataset(X_train, y_train), self.batch_size_, shuffle=True),
            "valid": DataLoader(TensorDataset(X_valid, y_valid), self.batch_size_, shuffle=True),
        }

        if self.use_quantile_encoding:
            bins = compute_bins(X=X_train, n_bins=self.max_bins)
            total_n_bins = sum(len(b) - 1 for b in bins)
            self._model: nn.Module = nn.Sequential(
                PiecewiseLinearEncoding(bins),
                _TorchMLP(
                    dim_in=total_n_bins,
                    dim_out=len(self.classes_),
                    dim_hid=self.dim_hid,
                    n_hid=self.n_hid,
                    dropout=self.dropout,
                ),
            )
        else:
            self._model = _TorchMLP(
                dim_in=self.n_features_in_,
                dim_out=len(self.classes_),
                dim_hid=self.dim_hid,
                n_hid=self.n_hid,
                dropout=self.dropout,
            )

        # --- Train model
        self.acc_history_, self.n_iter_ = _train_clf(
            model=self._model,
            device=self.device_,
            dataloaders=dataloaders,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_iter=self.max_iter,
            tol=self.tol,
            n_iter_no_change=self.n_iter_no_change,
            early_stopping=self.early_stopping,
            verbose=self.verbose,
        )

        return self

    def predict(self, X) -> NDArray:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, accept_sparse=False)
        X = np.array(X)

        self._model.to(self.device_)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(torch.tensor(X).float().to(self.device_))
            logits = logits.to("cpu").numpy()
            labels = np.argmax(logits, axis=1)

        return self._label_encoder.inverse_transform(labels)


if __name__ == "__main__":
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(MLPClassifier(max_bins=4, validation_fraction=0.5))

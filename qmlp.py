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

from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import Interval, StrOptions
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


class QMLPClassifier(ClassifierMixin, BaseEstimator):
    """Multilayer Perceptron classifier with optional Quantile Piecewise Linear Embedding.

    This class implements the MLP classifier with optional embedding layer based on quantile binning
    and piecewise linear embedding proposed in [1].

    Parameters
    ----------
    dim_hid : int, default=256,
        Size of each hidden layer of the MLP. For simplicity we assume that every hidden layer has
        the same size.

    dim_emb : int, default=8,
        Size of the embedding of *each* feature in the Qunantile PLE. The output of the embedding
        layer has shape `(n_samples, n_features * dim_emb)`. Only effective when
        `use_quantile_embedding=True`.

    n_hid : int, default=2,
        Number of hidden layers. The total number of layers in the MLP is `n_hid + 2`.
        Additionally there are `n_features` linear layers of size `(n_bins, dim_emb)` in the
        embedding layer.

    n_bins : int, default=48,
        Number of bins to use when discretizing the features based on quantiles for the Quantile
        Piecewise Linear Embedding. Only effective when `use_quantile_embedding=True`.

    dropout : float, default=0.2,
        Dropout probability.

    lr : float, default=3e-4,
        Learning rate.

    weight_decay : float, default=0.01,
        Strength of the L2 regularization term used in the AdamW optimizer.

    max_iter : int, default=200,
        Maximum number of iterations. The optimizer iterates until convergence (determined by `tol`)
        or this number of iterations

    batch_size : int, default=256,
        Size of minibatches.

    validation_fraction : float, default=0.2,
        The proportion of training data to set aside as validation set for early stopping. Must be
        between 0 and 1.

    tol : float, default=1e-4,
        Tolerance for the optimization. When the loss is not improving by at least `tol` for
        `n_iter_no_change` consecutive iterations, convergence is considered to be reached and
        training stops.

    n_iter_no_change : int, default=40,
        Maximum number of epochs to not meet tol improvement.

    early_stopping : bool, default=True,
        Whether to use early stopping to terminate training when validation score is not improving.
        If set to true, it will terminate training when validation score is not improving by at
        least `tol` for `n_iter_no_change` consecutive epochs

    use_quantile_embedding : bool, default=True,
        Whether to user Quantile Piecewise Linear Embedding layer described in [1]. If set to `True`
        it will first compute the appropriate bins on the training set (after setting aside the
        validation set) and then add embedding layer to the model.

    random_state : int, RandomState instance, default=None,
        Determines random number generation for weights and bias initialization, train-test split,
        and batch sampling. Pass an int for reproducible results across multiple function calls.

    verbose : bool, default=False,
        Whether to print progress messages to stdout.

    device : {"cpu", "cuda", "mps", "auto"}, default="auto",
        `torch.device` which is used for inference and training. If "auto" then the device is chosen
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
    | [Quantile Piecewise Linear Embedding(n_bins, dim_emb)]
    *****
        |
    [Flatten(start_dim=1)]
        |
    [Linear(dim_emb * n_features, dim_hid)]
        |
    [Batchnorm(dim_hid)]
        |
    [ReLU]
        |
    [Dropout]
        |
    ***** n_hid x
    | [Linear(dim_hid, dim_hid)]
    |     |
    | [Batchnorm(dim_hid)]
    |     |
    | [ReLU]
    |     |
    | [Dropout]
    *****
        |
    [Linear(dim_hid, n_classes)]
        |
    [Output(n_samples, n_classes)]
    ```

    Notes
    -----
    This class uses PyTorch library to implement and train the network itself. Additionally, for the
    Quantile Piecewise Linear Embedding layer we use the code published by the authors of the paper
    [1] which is available here: https://github.com/yandex-research/rtdl-num-embeddings.

    TODO: Write about philosophy of sane defaults -- not too much tweaks!!!

    References
    ----------
    [1] Yury Gorishniy, Ivan Rubachev, & Artem Babenko. (2023). On Embeddings for Numerical Features
        in Tabular Deep Learning. https://arxiv.org/pdf/2203.05556

    """

    _parameter_constraints = {
        "dim_hid": [Interval(Integral, 1, None, closed="left")],
        "dim_emb": [Interval(Integral, 1, None, closed="left")],
        "n_hid": [Interval(Integral, 1, None, closed="left")],
        "n_bins": [Interval(Integral, 1, None, closed="neither")],
        "dropout": [Interval(Real, 0, 1, closed="left")],
        "learning_rate": [Interval(Real, 0, None, closed="left")],
        "weight_decay": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "validation_fraction": [Interval(Real, 0, 1, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "n_iter_no_change": [Interval(Integral, 1, None, closed="left")],
        "use_quantile_embedding": ["boolean"],
        "random_state": ["random_state"],
        "verbose": ["boolean"],
        "early_stopping": ["boolean"],
        "device": [StrOptions({"cpu", "cuda", "mps", "auto"})],
    }

    def __init__(
        self,
        dim_hid: int = 256,
        dim_emb: int = 8,
        n_hid: int = 2,
        n_bins: int = 48,
        dropout: float = 0.2,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        max_iter: int = 200,
        batch_size: int = 256,
        validation_fraction: float = 0.2,
        tol: float = 1e-4,
        n_iter_no_change: int = 40,
        early_stopping: bool = True,
        use_quantile_embedding: bool = True,
        random_state: Optional[int | RandomState] = None,
        verbose: bool = False,
        device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
    ):
        super().__init__()

        self.dim_hid: int = dim_hid
        self.dim_emb: int = dim_emb
        self.n_hid: int = n_hid
        self.n_bins: int = n_bins

        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_decay
        self.dropout: float = dropout
        self.max_iter: int = max_iter
        self.batch_size: int = batch_size
        self.validation_fraction: float = validation_fraction
        self.tol: float = tol
        self.n_iter_no_change: int = n_iter_no_change
        self.early_stopping: bool = early_stopping
        self.use_quantile_embedding: bool = use_quantile_embedding

        self.random_state: Optional[int | RandomState] = random_state
        self.device: Literal["cpu", "cuda", "mps", "auto"] = device
        self.verbose: bool = verbose

    def fit(self, X, y):
        X, y = validate_data(self, X, y, accept_sparse=False, reset=True)
        check_classification_targets(y)
        X, y = np.array(X), np.array(y)

        self._label_encoder = LabelEncoder().fit(y)
        self.classes_ = self._label_encoder.classes_

        self._random_state: RandomState = check_random_state(self.random_state)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=self.validation_fraction,
            random_state=self._random_state,
            shuffle=True,
            stratify=y,
        )

        y_train = self._label_encoder.transform(y_train)
        y_valid = self._label_encoder.transform(y_valid)

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

        if self.use_quantile_embedding:
            self._model: nn.Module = nn.Sequential(
                PiecewiseLinearEmbeddings(
                    bins=compute_bins(X=X_train, n_bins=self.n_bins),
                    d_embedding=self.dim_emb,
                    # We use the default version recommended in the README here:
                    # https://github.com/yandex-research/rtdl-num-embeddings/tree/main/package#piecewise-linear-encoding--embeddings
                    activation=False,
                    version="B",
                ),
                nn.Flatten(start_dim=1),
                _TorchMLP(
                    dim_in=self.dim_emb * self.n_features_in_,
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

    check_estimator(QMLPClassifier(n_bins=4))

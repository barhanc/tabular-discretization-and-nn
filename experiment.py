from numpy.random import RandomState
from typing import Callable, Any, Literal
from sklearn.base import BaseEstimator


class Experiment:
    def __init__(
        self,
        name: str,
        datasets: dict[str, dict],
        model_parameters: dict[str, Any],
        model_constructors: dict[str, Callable[[Any, int | RandomState], BaseEstimator]],
        random_state: int | RandomState,
        task: Literal["classification", "regression"],
    ):
        pass

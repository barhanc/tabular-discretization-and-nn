import time
import pickle
import logging

import pandas as pd

from warnings import simplefilter
from collections import defaultdict

from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, KBinsDiscretizer

from qmlp import QMLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


datasets = {
    "electricity": 44120,
    "covertype": 44121,
    "pol": 44122,
    "house_16H": 44123,
    "kdd": 44124,
    "MagicTelescope": 44125,
    "bank_marketing": 44126,
    "phoneme": 44127,
    "miniboone": 44128,
    "higgs": 44129,
    "eye_movements": 44130,
    "jannis": 44131,
    "credit": 44089,
    "california": 44090,
}


logger = logging.getLogger()
logging.basicConfig(
    filename=f"logs/experiment_{time.strftime('%Y%m%d-%H%M%S')}.log",
    format="%(asctime)s:%(levelname)s:%(message)s",
    encoding="utf-8",
    filemode="w",
    force=True,
    level=logging.DEBUG,
)


random_state = 0
n_splits = 10
results = defaultdict(list)
simplefilter("ignore", UserWarning)


for dataset, data_id in datasets.items():
    X: pd.DataFrame
    y: pd.DataFrame
    X, y = fetch_openml(data_id=data_id, return_X_y=True, as_frame=True)

    logger.info(f"{dataset=} | size = {X.shape} | #NaN = {X.isna().sum().sum()}\n" + "=" * 100)

    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_idx, (train_idx, test_idx) in enumerate(skfold.split(X, y)):
        logger.info(f"Fold {fold_idx}:\n" + "-" * 100)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clfs = {
            "LogisticRegression": make_pipeline(
                QuantileTransformer(output_distribution="normal", random_state=random_state),
                LogisticRegression(max_iter=1_000, random_state=random_state),
            ),
            "RandomForestClassifier": RandomForestClassifier(random_state=random_state),
            "HistGradientBoostingClassifier": HistGradientBoostingClassifier(random_state=random_state),
            "MLPWithRawFeatures": make_pipeline(
                QuantileTransformer(output_distribution="normal", random_state=random_state),
                QMLPClassifier(dropout=0.2, use_quantile_encoding=False, random_state=random_state),
            ),
            "MLPWithQuantilePLEEncoding": QMLPClassifier(use_quantile_encoding=True, random_state=random_state),
            "MLPWithQuantileOrdinalEncoding": make_pipeline(
                KBinsDiscretizer(n_bins=48, encode="ordinal", strategy="quantile", random_state=random_state),
                QMLPClassifier(dropout=0.2, use_quantile_encoding=False, random_state=random_state),
            ),
        }

        try:
            for clf_name, clf in clfs.items():
                t = time.perf_counter()
                clf.fit(X_train, y_train)
                t = time.perf_counter() - t
                acc = accuracy_score(y_test, clf.predict(X_test))

                results[dataset, clf_name, "fit_times"].append(t)
                results[dataset, clf_name, "test_acc_scores"].append(acc)

                logger.info(f"{clf_name:>30s} | TEST ACC. {acc:.2%} | FIT TIME {t:.3f}s")

            fname = f"results/checkpoints/{time.strftime('%Y%m%d-%H%M%S')}_{dataset}_{fold_idx}.pickle"
            with open(fname, "wb") as file:
                pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)

        except Exception as e:
            logger.error(f"Exception at {dataset=}, {fold_idx=}: {e}")

logger.handlers.clear()

with open(f"results/results_{time.strftime('%Y%m%d-%H%M%S')}.pickle", "wb") as file:
    pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)

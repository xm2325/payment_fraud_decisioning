from __future__ import annotations
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_logistic(X, y):
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", C=0.5)),
    ])
    return model.fit(X, y)


def fit_lightgbm(X, y, sample_weight=None, n_estimators=350):
    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.045,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        class_weight="balanced",
        random_state=42,
        n_jobs=2,
        verbosity=-1,
    )
    return model.fit(X, y, sample_weight=sample_weight)


def fit_sigmoid_calibrator(raw_prob, y):
    p = np.clip(np.asarray(raw_prob), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    cal = LogisticRegression(C=1e6, max_iter=200)
    cal.fit(logit, y)
    return cal


def calibrate(cal, raw_prob):
    p = np.clip(np.asarray(raw_prob), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    return cal.predict_proba(logit)[:, 1]

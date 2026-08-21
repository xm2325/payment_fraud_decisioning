from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

NOVEL_FEATURES = [
    "log_amount", "sender_tx_1h", "sender_tx_24h", "recipient_fanin_24h",
    "device_activity_24h", "sender_unique_recipients_24h",
    "recipient_unique_senders_24h", "device_unique_senders_24h", "pair_tx_24h",
    "amount_vs_7d_mean", "balance_fraction"
]
TAIL_FEATURES = [
    "sender_tx_1h", "sender_tx_24h", "recipient_fanin_24h", "device_activity_24h",
    "sender_unique_recipients_24h", "recipient_unique_senders_24h",
    "device_unique_senders_24h"
]


def fit_novelty(train_df):
    scaler = RobustScaler()
    X = scaler.fit_transform(train_df[NOVEL_FEATURES])
    model = IsolationForest(n_estimators=250, contamination=0.01, random_state=42, n_jobs=2)
    model.fit(X)
    return scaler, model


def anomaly_score(scaler, model, df):
    X = scaler.transform(df[NOVEL_FEATURES])
    return -model.score_samples(X)


def fit_tail_detector(train_legit):
    """Fit an interpretable upper-tail detector without fraud labels.

    Each velocity feature is log-transformed and scaled by its historical 99th percentile.
    The final score is the largest scaled feature value.
    """
    scales = {}
    for c in TAIL_FEATURES:
        z = np.log1p(train_legit[c].to_numpy(float))
        q = float(np.quantile(z, 0.99))
        scales[c] = max(q, 1e-6)
    return scales


def tail_score(scales, df):
    scores = []
    for c in TAIL_FEATURES:
        scores.append(np.log1p(df[c].to_numpy(float)) / scales[c])
    return np.max(np.vstack(scores), axis=0)


def threshold_at_legit_fpr(scores, y, fpr=0.01):
    legit = np.asarray(scores)[np.asarray(y) == 0]
    return float(np.quantile(legit, 1 - fpr))

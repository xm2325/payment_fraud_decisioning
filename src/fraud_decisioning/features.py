from __future__ import annotations
from collections import Counter, defaultdict, deque
import numpy as np
import pandas as pd


def _evict_pair_queue(q, counts, cutoff):
    while q and q[0][0] < cutoff:
        _, key = q.popleft()
        counts[key] -= 1
        if counts[key] <= 0:
            del counts[key]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build strictly backward-looking transaction and network features.

    Every rolling feature is read before the current transaction is appended.
    This gives point-in-time semantics suitable for fraud decisioning.
    """
    x = df.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
    n = len(x)
    # Pandas 3 may preserve microsecond datetime resolution; normalize explicitly.
    ts = (x["timestamp"].dt.as_unit("ns").astype("int64") // 10**9).to_numpy()
    senders = x["sender"].to_numpy()
    recipients = x["recipient"].to_numpy()
    devices = x["device_id"].to_numpy()
    amounts = x["amount"].to_numpy(float)

    sender_1h = defaultdict(deque)
    sender_24h = defaultdict(deque)
    sender_7d = defaultdict(deque)
    sender_24h_sum = defaultdict(float)
    sender_7d_sum = defaultdict(float)
    recipient_24h = defaultdict(deque)
    device_24h = defaultdict(deque)

    # Network-style rolling state.
    sender_recipient_q = defaultdict(deque)
    sender_recipient_counts = defaultdict(Counter)
    recipient_sender_q = defaultdict(deque)
    recipient_sender_counts = defaultdict(Counter)
    device_sender_q = defaultdict(deque)
    device_sender_counts = defaultdict(Counter)
    pair_24h = defaultdict(deque)

    f_sender_1h = np.zeros(n, dtype=np.float32)
    f_sender_24h = np.zeros(n, dtype=np.float32)
    f_sender_amt_24h = np.zeros(n, dtype=np.float32)
    f_sender_mean_7d = np.zeros(n, dtype=np.float32)
    f_recipient_24h = np.zeros(n, dtype=np.float32)
    f_device_24h = np.zeros(n, dtype=np.float32)
    f_sender_unique_recipients_24h = np.zeros(n, dtype=np.float32)
    f_recipient_unique_senders_24h = np.zeros(n, dtype=np.float32)
    f_device_unique_senders_24h = np.zeros(n, dtype=np.float32)
    f_pair_tx_24h = np.zeros(n, dtype=np.float32)

    H1, H24, D7 = 3600, 86400, 7 * 86400
    # Process equal-timestamp transactions as one batch. No transaction at time t
    # may use another event at the same recorded timestamp as historical state.
    # This matches the strict event-time SQL semantics used in the repo.
    i = 0
    while i < n:
        t = ts[i]
        j = i + 1
        while j < n and ts[j] == t:
            j += 1

        # First read historical state for every transaction at this timestamp.
        for k in range(i, j):
            s, r, d, a = senders[k], recipients[k], devices[k], amounts[k]
            q1 = sender_1h[s]
            while q1 and q1[0] < t - H1:
                q1.popleft()
            q24 = sender_24h[s]
            while q24 and q24[0][0] < t - H24:
                _, old_a = q24.popleft(); sender_24h_sum[s] -= old_a
            q7 = sender_7d[s]
            while q7 and q7[0][0] < t - D7:
                _, old_a = q7.popleft(); sender_7d_sum[s] -= old_a
            qr = recipient_24h[r]
            while qr and qr[0] < t - H24:
                qr.popleft()
            qd = device_24h[d]
            while qd and qd[0] < t - H24:
                qd.popleft()

            srq, src = sender_recipient_q[s], sender_recipient_counts[s]
            _evict_pair_queue(srq, src, t - H24)
            rsq, rsc = recipient_sender_q[r], recipient_sender_counts[r]
            _evict_pair_queue(rsq, rsc, t - H24)
            dsq, dsc = device_sender_q[d], device_sender_counts[d]
            _evict_pair_queue(dsq, dsc, t - H24)
            pq = pair_24h[(s, r)]
            while pq and pq[0] < t - H24:
                pq.popleft()

            f_sender_1h[k] = len(q1)
            f_sender_24h[k] = len(q24)
            f_sender_amt_24h[k] = sender_24h_sum[s]
            f_sender_mean_7d[k] = sender_7d_sum[s] / len(q7) if q7 else a
            f_recipient_24h[k] = len(qr)
            f_device_24h[k] = len(qd)
            f_sender_unique_recipients_24h[k] = len(src)
            f_recipient_unique_senders_24h[k] = len(rsc)
            f_device_unique_senders_24h[k] = len(dsc)
            f_pair_tx_24h[k] = len(pq)

        # Only after all reads at t do we update history with the t events.
        for k in range(i, j):
            s, r, d, a = senders[k], recipients[k], devices[k], amounts[k]
            sender_1h[s].append(t)
            sender_24h[s].append((t, a)); sender_24h_sum[s] += a
            sender_7d[s].append((t, a)); sender_7d_sum[s] += a
            recipient_24h[r].append(t)
            device_24h[d].append(t)
            sender_recipient_q[s].append((t, r)); sender_recipient_counts[s][r] += 1
            recipient_sender_q[r].append((t, s)); recipient_sender_counts[r][s] += 1
            device_sender_q[d].append((t, s)); device_sender_counts[d][s] += 1
            pair_24h[(s, r)].append(t)
        i = j

    x["log_amount"] = np.log1p(x["amount"])
    x["balance_fraction"] = (x["amount"] / x["old_balance_sender"].clip(lower=1)).clip(0, 20)
    x["hour"] = x["timestamp"].dt.hour
    x["is_night"] = x["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    x["sender_tx_1h"] = f_sender_1h
    x["sender_tx_24h"] = f_sender_24h
    x["sender_amount_24h"] = f_sender_amt_24h
    x["sender_mean_amount_7d"] = f_sender_mean_7d
    x["recipient_fanin_24h"] = f_recipient_24h
    x["device_activity_24h"] = f_device_24h
    x["sender_unique_recipients_24h"] = f_sender_unique_recipients_24h
    x["recipient_unique_senders_24h"] = f_recipient_unique_senders_24h
    x["device_unique_senders_24h"] = f_device_unique_senders_24h
    x["pair_tx_24h"] = f_pair_tx_24h
    x["amount_vs_7d_mean"] = x["amount"] / np.maximum(x["sender_mean_amount_7d"], 1.0)

    type_dummies = pd.get_dummies(x["type"], prefix="type", dtype=int)
    x = pd.concat([x, type_dummies], axis=1)
    feature_cols = [
        "log_amount", "balance_fraction", "account_age_days", "device_change",
        "country_mismatch", "recipient_new", "hour", "is_night", "sender_tx_1h",
        "sender_tx_24h", "sender_amount_24h", "recipient_fanin_24h",
        "device_activity_24h", "sender_unique_recipients_24h",
        "recipient_unique_senders_24h", "device_unique_senders_24h", "pair_tx_24h",
        "amount_vs_7d_mean",
    ] + sorted(type_dummies.columns.tolist())
    return x, feature_cols

from __future__ import annotations

TRANSACTION_ONLY_FEATURES = [
    "log_amount",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]

BEHAVIOURAL_FEATURES = TRANSACTION_ONLY_FEATURES + [
    "sender_tx_1h",
    "sender_tx_24h",
    "sender_amount_24h",
    "recipient_fanin_24h",
    "amount_vs_7d_mean",
]

RELATIONAL_FEATURES = BEHAVIOURAL_FEATURES + [
    "sender_tx_7d",
    "recipient_tx_7d",
    "recipient_amount_24h",
    "pair_tx_7d",
    "pair_amount_7d",
    "sender_recipient_share_7d",
    "sender_unique_recipients_7d",
    "recipient_unique_senders_7d",
]

SIMULATOR_BALANCE_FEATURES = [
    "balance_fraction",
    "orig_balance_delta",
    "dest_balance_delta",
]

FEATURE_COLUMNS = RELATIONAL_FEATURES + SIMULATOR_BALANCE_FEATURES

FEATURE_SETS = {
    "transaction_only": TRANSACTION_ONLY_FEATURES,
    "transaction_plus_history": BEHAVIOURAL_FEATURES,
    "transaction_plus_relational": RELATIONAL_FEATURES,
    "full_with_simulator_balances": FEATURE_COLUMNS,
}

BALANCE_FREE_CANDIDATES = [
    "transaction_only",
    "transaction_plus_history",
    "transaction_plus_relational",
]

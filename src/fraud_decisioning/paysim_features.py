from __future__ import annotations

FEATURE_COLUMNS = [
    "log_amount",
    "balance_fraction",
    "sender_tx_1h",
    "sender_tx_24h",
    "sender_amount_24h",
    "recipient_fanin_24h",
    "amount_vs_7d_mean",
    "orig_balance_delta",
    "dest_balance_delta",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]

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

FEATURE_SETS = {
    "transaction_only": TRANSACTION_ONLY_FEATURES,
    "transaction_plus_history": BEHAVIOURAL_FEATURES,
    "full_with_simulator_balances": FEATURE_COLUMNS,
}

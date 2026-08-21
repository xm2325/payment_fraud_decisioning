-- Point-in-time recipient investigation features.
-- Same-step transactions are simultaneous: the current hour cannot see itself.
WITH base AS (
    SELECT
        step,
        nameOrig AS sender,
        nameDest AS recipient,
        type,
        amount
    FROM transactions
),
features AS (
    SELECT
        *,
        COUNT(*) OVER (
            PARTITION BY recipient ORDER BY step
            RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING
        ) AS recipient_tx_24h,
        SUM(amount) OVER (
            PARTITION BY recipient ORDER BY step
            RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING
        ) AS recipient_amount_24h,
        APPROX_COUNT_DISTINCT(sender) OVER (
            PARTITION BY recipient ORDER BY step
            RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING
        ) AS recipient_unique_senders_7d,
        COUNT(*) OVER (
            PARTITION BY sender, recipient ORDER BY step
            RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING
        ) AS pair_tx_7d
    FROM base
)
SELECT
    step,
    sender,
    recipient,
    type,
    amount,
    COALESCE(recipient_tx_24h, 0) AS recipient_tx_24h,
    COALESCE(recipient_amount_24h, 0) AS recipient_amount_24h,
    COALESCE(recipient_unique_senders_7d, 0) AS recipient_unique_senders_7d,
    COALESCE(pair_tx_7d, 0) AS pair_tx_7d
FROM features;

-- Retrospective recipient concentration / mule-screening investigation.
-- Fraud labels are only used in the final audit columns, not in the historical features.
WITH base AS (
    SELECT
        step,
        nameOrig AS sender,
        nameDest AS recipient,
        amount,
        type,
        isFraud AS is_fraud
    FROM transactions
),
features AS (
    SELECT
        *,
        COUNT(*) OVER (
            PARTITION BY recipient ORDER BY step
            RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING
        ) AS recipient_tx_24h,
        COALESCE(SUM(amount) OVER (
            PARTITION BY recipient ORDER BY step
            RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING
        ), 0) AS recipient_amount_24h,
        APPROX_COUNT_DISTINCT(sender) OVER (
            PARTITION BY recipient ORDER BY step
            RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING
        ) AS unique_senders_7d,
        COUNT(*) OVER (
            PARTITION BY sender, recipient ORDER BY step
            RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING
        ) AS pair_tx_7d
    FROM base
),
ranked AS (
    SELECT
        *,
        NTILE(100) OVER (
            ORDER BY recipient_amount_24h DESC, unique_senders_7d DESC, recipient_tx_24h DESC
        ) AS investigation_percentile
    FROM features
)
SELECT
    investigation_percentile,
    COUNT(*) AS transactions,
    AVG(is_fraud) AS retrospective_fraud_rate,
    SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) AS retrospective_fraud_value,
    AVG(recipient_tx_24h) AS avg_recipient_tx_24h,
    AVG(unique_senders_7d) AS avg_unique_senders_7d,
    AVG(pair_tx_7d) AS avg_pair_tx_7d
FROM ranked
WHERE investigation_percentile <= 10
GROUP BY investigation_percentile
ORDER BY investigation_percentile;

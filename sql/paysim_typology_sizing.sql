-- Offline fraud sizing on labelled PaySim data.
-- Labels are used for retrospective sizing only, not as online features.
WITH labelled AS (
    SELECT
        step,
        type,
        amount,
        isFraud AS is_fraud,
        CASE
            WHEN amount < 1e4 THEN '<10k'
            WHEN amount < 5e4 THEN '10k-50k'
            WHEN amount < 2e5 THEN '50k-200k'
            ELSE '>=200k'
        END AS amount_band
    FROM transactions
),
summary AS (
    SELECT
        type,
        amount_band,
        COUNT(*) AS transactions,
        SUM(is_fraud) AS fraud_transactions,
        SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) AS fraud_value,
        SUM(amount) AS total_value
    FROM labelled
    GROUP BY 1, 2
)
SELECT
    type,
    amount_band,
    transactions,
    fraud_transactions,
    fraud_transactions * 1.0 / NULLIF(transactions, 0) AS fraud_rate,
    fraud_value,
    fraud_value / NULLIF(SUM(fraud_value) OVER (), 0) AS share_of_fraud_value,
    total_value
FROM summary
ORDER BY fraud_value DESC;

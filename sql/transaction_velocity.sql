-- PostgreSQL-style point-in-time velocity features.
-- Current-row data are excluded, preventing target-time leakage.
SELECT
    transaction_id,
    sender,
    event_ts,
    amount,
    COUNT(*) OVER (
        PARTITION BY sender ORDER BY event_ts
        RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
    ) AS sender_tx_1h,
    COUNT(*) OVER (
        PARTITION BY sender ORDER BY event_ts
        RANGE BETWEEN INTERVAL '24 hour' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
    ) AS sender_tx_24h,
    SUM(amount) OVER (
        PARTITION BY sender ORDER BY event_ts
        RANGE BETWEEN INTERVAL '24 hour' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
    ) AS sender_amount_24h
FROM transactions;

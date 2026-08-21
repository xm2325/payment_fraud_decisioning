-- Recipient pressure: number and value of incoming payments before each transaction.
SELECT
    transaction_id,
    recipient,
    event_ts,
    COUNT(*) OVER (
        PARTITION BY recipient ORDER BY event_ts
        RANGE BETWEEN INTERVAL '24 hour' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
    ) AS recipient_fanin_24h,
    SUM(amount) OVER (
        PARTITION BY recipient ORDER BY event_ts
        RANGE BETWEEN INTERVAL '24 hour' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
    ) AS recipient_value_24h
FROM transactions;

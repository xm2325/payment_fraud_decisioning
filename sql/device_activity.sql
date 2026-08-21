-- Shared-device activity is useful for account-farm and coordinated-attack investigation.
SELECT
    transaction_id,
    device_id,
    event_ts,
    COUNT(*) OVER (
        PARTITION BY device_id ORDER BY event_ts
        RANGE BETWEEN INTERVAL '24 hour' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
    ) AS device_activity_24h
FROM transactions;

-- Executable SQLite reference query used by tests/test_sql_parity.py.
-- event_ts is an integer Unix timestamp in seconds. Same-timestamp events are
-- excluded from one another, matching build_features() batch semantics.
SELECT
  t.transaction_id,
  (
    SELECT COUNT(*) FROM transactions h
    WHERE h.sender = t.sender
      AND h.event_ts < t.event_ts
      AND h.event_ts >= t.event_ts - 3600
  ) AS sender_tx_1h,
  (
    SELECT COUNT(*) FROM transactions h
    WHERE h.sender = t.sender
      AND h.event_ts < t.event_ts
      AND h.event_ts >= t.event_ts - 86400
  ) AS sender_tx_24h,
  (
    SELECT COUNT(*) FROM transactions h
    WHERE h.recipient = t.recipient
      AND h.event_ts < t.event_ts
      AND h.event_ts >= t.event_ts - 86400
  ) AS recipient_fanin_24h,
  (
    SELECT COUNT(*) FROM transactions h
    WHERE h.device_id = t.device_id
      AND h.event_ts < t.event_ts
      AND h.event_ts >= t.event_ts - 86400
  ) AS device_activity_24h
FROM transactions t
ORDER BY t.transaction_id;

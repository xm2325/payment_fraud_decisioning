-- Label-free recipient investigation queue for PaySim-like data.
-- All behavioural features are strict prior-step history: transactions in the
-- current hour cannot see one another. Fraud labels are intentionally absent
-- from candidate construction and should only be joined later for backtesting.
WITH base AS (
    SELECT
        step::INTEGER AS step,
        nameOrig::VARCHAR AS sender,
        nameDest::VARCHAR AS recipient,
        type::VARCHAR AS type,
        amount::DOUBLE AS amount,
        COUNT(*) OVER (
            PARTITION BY nameDest ORDER BY step
            RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING
        )::DOUBLE AS recipient_fanin_24h,
        COALESCE(SUM(amount) OVER (
            PARTITION BY nameDest ORDER BY step
            RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING
        ), 0)::DOUBLE AS recipient_amount_24h,
        COUNT(*) OVER (
            PARTITION BY nameDest ORDER BY step
            RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING
        )::DOUBLE AS recipient_tx_7d,
        APPROX_COUNT_DISTINCT(nameOrig) OVER (
            PARTITION BY nameDest ORDER BY step
            RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING
        )::DOUBLE AS recipient_unique_senders_7d,
        COUNT(*) OVER (
            PARTITION BY nameOrig, nameDest ORDER BY step
            RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING
        )::DOUBLE AS pair_tx_7d
    FROM transactions
), ranked AS (
    SELECT
        *,
        PERCENT_RANK() OVER (PARTITION BY step ORDER BY recipient_fanin_24h) AS fanin_pct,
        PERCENT_RANK() OVER (PARTITION BY step ORDER BY recipient_amount_24h) AS recipient_amount_pct,
        PERCENT_RANK() OVER (PARTITION BY step ORDER BY recipient_tx_7d) AS recipient_tx_pct,
        PERCENT_RANK() OVER (PARTITION BY step ORDER BY recipient_unique_senders_7d) AS unique_senders_pct
    FROM base
), scored AS (
    SELECT
        *,
        0.25 * fanin_pct
        + 0.25 * recipient_amount_pct
        + 0.15 * recipient_tx_pct
        + 0.35 * unique_senders_pct AS recipient_investigation_score
    FROM ranked
)
SELECT
    step,
    sender,
    recipient,
    type,
    amount,
    recipient_fanin_24h,
    recipient_amount_24h,
    recipient_tx_7d,
    recipient_unique_senders_7d,
    pair_tx_7d,
    recipient_investigation_score,
    CONCAT_WS('|',
        CASE WHEN unique_senders_pct >= 0.99 THEN 'MANY_PRIOR_SENDERS' END,
        CASE WHEN fanin_pct >= 0.99 THEN 'HIGH_PRIOR_FAN_IN' END,
        CASE WHEN recipient_amount_pct >= 0.99 THEN 'HIGH_PRIOR_RECIPIENT_VALUE' END,
        CASE WHEN pair_tx_7d = 0 THEN 'NEW_SENDER_RECIPIENT_PAIR' END
    ) AS investigation_reasons
FROM scored
ORDER BY recipient_investigation_score DESC, amount DESC;

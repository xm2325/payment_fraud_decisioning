-- Retrospective candidate-rule backtest.
-- The rule grid is intentionally small and interpretable; production rules should
-- be selected on a validation interval and checked out of time.
WITH candidate_rules AS (
    SELECT * FROM (VALUES
        ('transfer_50k',  'TRANSFER',  50000.0),
        ('transfer_100k', 'TRANSFER', 100000.0),
        ('cashout_50k',   'CASH_OUT',  50000.0),
        ('cashout_100k',  'CASH_OUT', 100000.0)
    ) AS r(rule_name, tx_type, min_amount)
),
scored AS (
    SELECT
        r.rule_name,
        t.isFraud AS is_fraud,
        t.amount,
        CASE WHEN t.type = r.tx_type AND t.amount >= r.min_amount THEN 1 ELSE 0 END AS alert
    FROM transactions t
    CROSS JOIN candidate_rules r
),
agg AS (
    SELECT
        rule_name,
        SUM(alert) AS alerts,
        SUM(CASE WHEN alert = 1 AND is_fraud = 1 THEN 1 ELSE 0 END) AS tp,
        SUM(CASE WHEN alert = 1 AND is_fraud = 0 THEN 1 ELSE 0 END) AS fp,
        SUM(is_fraud) AS fraud_n,
        SUM(CASE WHEN alert = 1 AND is_fraud = 1 THEN amount ELSE 0 END) AS fraud_value_captured,
        SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) AS fraud_value,
        SUM(CASE WHEN is_fraud = 0 THEN 1 ELSE 0 END) AS legit_n
    FROM scored
    GROUP BY rule_name
)
SELECT
    rule_name,
    alerts,
    tp * 1.0 / NULLIF(alerts, 0) AS precision,
    tp * 1.0 / NULLIF(fraud_n, 0) AS recall,
    fp * 1.0 / NULLIF(legit_n, 0) AS legitimate_flag_rate,
    fraud_value_captured / NULLIF(fraud_value, 0) AS fraud_value_recall
FROM agg
ORDER BY fraud_value_recall DESC, legitimate_flag_rate ASC;

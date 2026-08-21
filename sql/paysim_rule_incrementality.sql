-- Compare a validation-selected simple amount/type rule with a model decision table.
-- Expected tables:
--   transactions(step, type, amount, isFraud)
--   model_scores(step, transaction_id, score)
-- The thresholds below should be supplied from validation, never tuned on test.
-- :rule_log_amount_threshold and :model_threshold are bind parameters.
WITH test AS (
    SELECT
        t.*,
        m.score,
        CASE
            WHEN t.type IN ('TRANSFER', 'CASH_OUT')
             AND LN(1 + GREATEST(t.amount, 0)) >= :rule_log_amount_threshold
            THEN 1 ELSE 0
        END AS rule_alert,
        CASE WHEN m.score >= :model_threshold THEN 1 ELSE 0 END AS model_alert
    FROM transactions t
    JOIN model_scores m USING (transaction_id)
    WHERE t.step >= :test_start_step
),
long AS (
    SELECT 'simple_amount_type_rule' AS detector, isFraud AS is_fraud, amount, rule_alert AS alert FROM test
    UNION ALL
    SELECT 'ml_model', isFraud, amount, model_alert FROM test
),
agg AS (
    SELECT
        detector,
        SUM(alert) AS alerts,
        SUM(CASE WHEN alert = 1 AND is_fraud = 1 THEN 1 ELSE 0 END) AS tp,
        SUM(CASE WHEN alert = 1 AND is_fraud = 0 THEN 1 ELSE 0 END) AS fp,
        SUM(is_fraud) AS fraud_n,
        SUM(CASE WHEN is_fraud = 0 THEN 1 ELSE 0 END) AS legit_n,
        SUM(CASE WHEN alert = 1 AND is_fraud = 1 THEN amount ELSE 0 END) AS captured_fraud_value,
        SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) AS total_fraud_value
    FROM long
    GROUP BY detector
)
SELECT
    detector,
    alerts,
    tp * 1.0 / NULLIF(alerts, 0) AS precision,
    tp * 1.0 / NULLIF(fraud_n, 0) AS fraud_recall,
    fp * 1.0 / NULLIF(legit_n, 0) AS legitimate_flag_rate,
    captured_fraud_value / NULLIF(total_fraud_value, 0) AS fraud_value_recall
FROM agg
ORDER BY detector;

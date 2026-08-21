-- Threshold-drift diagnostic for a time-ordered fraud system.
-- If model_scores is unavailable, replace score with a candidate ranking signal.
WITH joined AS (
    SELECT
        t.step,
        t.isFraud AS is_fraud,
        t.amount,
        s.score,
        CASE
            WHEN t.step < :validation_start_step THEN 'train'
            WHEN t.step < :test_start_step THEN 'validation'
            ELSE 'future_test'
        END AS period
    FROM transactions t
    JOIN model_scores s USING (transaction_id)
),
quantiles AS (
    SELECT
        period,
        COUNT(*) AS n,
        AVG(is_fraud) AS fraud_rate,
        AVG(score) AS mean_score,
        APPROX_QUANTILE(score, 0.50) AS score_p50,
        APPROX_QUANTILE(score, 0.90) AS score_p90,
        APPROX_QUANTILE(score, 0.99) AS score_p99,
        APPROX_QUANTILE(amount, 0.99) AS amount_p99
    FROM joined
    GROUP BY period
),
threshold_perf AS (
    SELECT
        period,
        SUM(CASE WHEN score >= :model_threshold THEN 1 ELSE 0 END) AS alerts,
        SUM(CASE WHEN score >= :model_threshold AND is_fraud = 1 THEN 1 ELSE 0 END) AS tp,
        SUM(CASE WHEN score >= :model_threshold AND is_fraud = 0 THEN 1 ELSE 0 END) AS fp,
        SUM(is_fraud) AS fraud_n,
        SUM(CASE WHEN is_fraud = 0 THEN 1 ELSE 0 END) AS legit_n
    FROM joined
    GROUP BY period
)
SELECT
    q.*,
    p.alerts,
    p.tp * 1.0 / NULLIF(p.fraud_n, 0) AS recall_at_locked_threshold,
    p.fp * 1.0 / NULLIF(p.legit_n, 0) AS legit_flag_rate_at_locked_threshold
FROM quantiles q
JOIN threshold_perf p USING (period)
ORDER BY CASE period WHEN 'train' THEN 1 WHEN 'validation' THEN 2 ELSE 3 END;

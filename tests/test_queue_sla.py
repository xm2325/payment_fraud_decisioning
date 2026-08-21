import numpy as np
import pandas as pd
from fraud_decisioning.queue_sla import queue_sla_stress


def test_queue_backlog_grows_when_arrivals_exceed_capacity():
    n = 24
    df = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=n, freq="h")})
    p = np.full(n, 0.10)
    a = np.zeros(n)
    out = queue_sla_stress(df, p, a, 0.03, 0.25, 1.0,
                           analyst_capacities_per_hour=(1.0,),
                           volume_multipliers=(1.0, 2.0))
    r1 = out[out.traffic_multiplier.eq(1.0)].iloc[0]
    r2 = out[out.traffic_multiplier.eq(2.0)].iloc[0]
    assert r1.final_backlog_cases == 0
    assert r2.final_backlog_cases > 0

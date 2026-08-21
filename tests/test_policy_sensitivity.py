import numpy as np
from fraud_decisioning.policy_sensitivity import policy_sensitivity


def test_policy_sensitivity_selects_validation_policy_only():
    yv = np.array([0,0,0,1,1,0,1,0,0,1])
    pv = np.array([.01,.05,.2,.7,.9,.1,.6,.03,.15,.8])
    av = np.array([10,20,10,100,200,30,150,10,20,120], dtype=float)
    out = policy_sensitivity(yv,pv,av,yv,pv,av)
    assert len(out) == 4
    assert (out.test_policy_regret >= -1e-12).all()

from fraud_decisioning.api import decision, RiskRequest

def test_decision_boundaries():
    assert decision(RiskRequest(risk_probability=0.1, amount=10))["action"] == "approve"
    assert decision(RiskRequest(risk_probability=0.3, amount=10))["action"] == "review"
    assert decision(RiskRequest(risk_probability=0.9, amount=10))["action"] == "block"

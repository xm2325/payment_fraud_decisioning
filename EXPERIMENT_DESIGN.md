# Fraud intervention experiment design

The public/synthetic transaction data do not contain randomized intervention assignment or customer-completion events. The project therefore does **not** report a fabricated A/B treatment effect.

The pipeline instead uses the validation-period calibrated risk score to define the operational review band. It measures the observed fraud rate in that eligible population, specifies a 25% relative reduction as the target effect, and calculates the approximate sample size required per randomized arm at two-sided alpha 0.05 and 80% power.

A real production test would randomize an intervention such as step-up verification within that risk band. The primary endpoint could be confirmed fraud incidence or fraud loss per eligible transaction. Customer completion, abandonment, support-contact rate, and false-positive burden should be pre-registered as guardrails. Those guardrails require product event data that are not available here.

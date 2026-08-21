# Fraud mitigation hand-off

This project separates three decisions that are often mixed together: risk scoring, intervention choice, and investigation priority. A model score is not itself a mitigation.

For a high supervised fraud score with a known typology, the default path is loss-oriented exploitation: block only above the validated block threshold, send the intermediate band to review, and attach reason codes such as high model risk, amount tail, device change, or recipient activity. For a behavioural-tail or relational anomaly with a weak supervised score, the default path is exploration: reserve analyst capacity, avoid automatic blocking unless an independent rule supports it, and use confirmed cases as matured labels for later retraining.

Operational monitoring covers model score distribution, legitimate flag rate, fraud-value capture, alert arrival rate, queue backlog, and exploration share. If the review queue becomes capacity constrained, the adaptive router raises the admission cutoff while retaining a token-based exploration budget so that novel-fraud discovery is not silently reduced to zero.

The full PaySim benchmark adds one extra control: simulator-derived balance fields are excluded from the deployment-style reference model because they produce unusually high separability. Balance-free transaction, history, and relational models are compared separately. A feature is kept only if it improves out-of-time value or investigation quality enough to justify its operational cost.

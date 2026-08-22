# Genomics England backend evidence plan

This branch strengthens existing backend engineering evidence without changing or overstating the fraud modelling claims in the main project.

Target capabilities to add and verify:

- FastAPI REST endpoints with explicit request/response schemas;
- PostgreSQL-compatible persistence behind a repository layer, with SQLite for local tests;
- role-based access control (RBAC) for researcher/reviewer/admin-style roles;
- JWT/OIDC-compatible bearer-token validation boundary with a local test issuer;
- audit-event persistence for access and decision actions;
- idempotent create/update semantics and useful HTTP status codes;
- pytest coverage for 401/403/404/409 paths, transaction rollback and audit logging;
- Docker/Compose local development path;
- AWS reference deployment contract using API Gateway/ECS or Lambda, RDS PostgreSQL, IAM, Secrets Manager and CloudWatch;
- CI checks that run unit and API tests.

The AWS material on this branch is a reference architecture unless an actual deployment is performed and verified. The CV must not claim production AWS deployment from this branch alone.

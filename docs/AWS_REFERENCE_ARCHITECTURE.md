# AWS reference architecture for the research-access API

This document maps the tested local FastAPI access-control service to an AWS deployment design. It is a reference architecture, not evidence that this repository has been deployed to AWS.

## Runtime mapping

```text
Client / researcher
        |
        | HTTPS + OIDC bearer token
        v
Amazon API Gateway
        |
        v
FastAPI service on ECS Fargate
        |
        +----------> Amazon RDS for PostgreSQL
        |
        +----------> Amazon CloudWatch logs/metrics

Identity provider (for example Okta or another OIDC IdP)
        |
        +----------> JWKS / issuer / audience validation

AWS Secrets Manager
        |
        +----------> database credentials / IdP configuration
```

An AWS Lambda deployment is also possible for the API layer, but ECS Fargate is the primary reference here because the existing application is an ASGI service and can retain a conventional container runtime.

## IAM boundaries

The application task role should have only the permissions required to:

- read the specific Secrets Manager secrets used by this service;
- write application logs and metrics to its CloudWatch resources;
- use any explicitly declared AWS service integrations.

Database access should be constrained by network controls and database credentials rather than broad IAM permissions. Human administrators and CI/CD roles should be separate from the runtime task role.

## OIDC production boundary

`access_control.py` uses a deterministic local token contract for tests. A production adapter should validate:

- signature against the IdP JWKS;
- issuer (`iss`);
- audience (`aud`);
- expiry (`exp`) and not-before (`nbf`);
- subject (`sub`);
- role/group claims mapped to application RBAC.

Authentication failure returns 401. An authenticated principal without the required role returns 403.

## RDS/PostgreSQL migration

The local store uses SQLite so CI remains self-contained. Production persistence should use PostgreSQL/RDS behind the same repository interface. Required migration checks include:

- unique request/idempotency constraint;
- explicit transaction boundaries;
- schema migration versioning;
- indexes for requester, status and audit-event lookup;
- rollback tests for failed writes;
- connection-pool limits and timeout handling.

## Operational controls

Before a real deployment, add:

- structured JSON logging with request/correlation IDs;
- CloudWatch alarms for 5xx rate, latency and database connection failures;
- health/readiness endpoints that distinguish process health from dependency readiness;
- secret rotation;
- database backups and recovery tests;
- infrastructure as code (Terraform or AWS CDK);
- dependency and container vulnerability scanning.

## Evidence boundary

Safe CV wording from the current branch:

> Implemented and tested a FastAPI research-access service with RBAC, transactional persistence, audit logging and explicit negative HTTP paths; documented an AWS reference deployment using API Gateway, ECS Fargate, RDS PostgreSQL, IAM, Secrets Manager and CloudWatch.

Do **not** replace "documented an AWS reference deployment" with "deployed to AWS" unless a real deployment is completed and verified.

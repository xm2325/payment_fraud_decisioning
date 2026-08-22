# Research-access API demo

This API is a small software-engineering extension to the fraud workbench. It is intentionally domain-neutral at the storage layer and uses synthetic dataset names in tests. Its purpose is to demonstrate access-control, persistence and API reliability patterns relevant to sensitive research environments.

## Local authentication contract

For local tests only, send a bearer token in this form:

```text
Authorization: Bearer <subject>|<role>[,<role>...]
```

Roles are `researcher`, `reviewer` and `admin`.

Examples:

```text
Bearer alice|researcher
Bearer bob|reviewer
Bearer service-admin|admin
```

This is not presented as production authentication. `AWS_REFERENCE_ARCHITECTURE.md` documents the OIDC/JWKS boundary that would replace it.

## Endpoints

```text
POST /research-access/requests
GET  /research-access/requests/{request_id}
POST /research-access/requests/{request_id}/review
GET  /research-access/audit-events
```

Key behaviours:

- missing/invalid authentication -> 401;
- authenticated principal without the required role -> 403;
- missing request -> 404;
- duplicate request or repeated review -> 409;
- invalid review decision -> 422;
- request creation and review both write audit events inside the same transaction.

## Test focus

`tests/test_research_access_api.py` verifies the negative HTTP paths, ownership checks, reviewer/admin separation, duplicate-request handling, single-review rule and audit-log visibility.

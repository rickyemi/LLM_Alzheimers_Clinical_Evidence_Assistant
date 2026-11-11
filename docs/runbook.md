# Runbook

## Start and verify

```bash
make index && make api
bash scripts/smoke_test_api.sh http://localhost:8000
```

## Alerts and responses

| Signal | Likely cause | Action |
|---|---|---|
| Ungrounded rate above 25% | Users asking outside the corpus | Review sample questions; refresh corpus or confirm scope; add to golden set |
| Mean best score drops 15% from baseline | Corpus stale or vocabulary shift | Re-index; re-run `make eval` |
| `AuditLog.verify` fails | Tampering or disk corruption | Freeze the service, preserve the file, open a security incident |
| Escalation spike | New unsafe pattern or tool outage | Inspect `artifacts/review_queue.jsonl`; add red-team cases |
| Unsafe answer reported | Guardrail gap | Disable agent endpoint if needed; add golden case; fix; re-run gates; update model card |

## Rollback

Redeploy the previous image tag. The index is baked into each image, so corpus and code roll back together.

## Release

1. `make lint test eval notebooks` all green.
2. Update `docs/model_card.md`.
3. Governance sign-off recorded.
4. `docker build`, deploy to staging, run the smoke test, then promote.

# Hardening

Current security and operational posture of this repository, followed by a staged ladder toward production use. Grounded in what the code is: a single-process, local batch pipeline for fine-tuning and evaluating translation models. It exposes no network service and handles no user data at runtime.

## Current posture

**Authentication and secrets**
- No credentials exist in the repository at HEAD (verified by pattern and entropy scan of all tracked files). There is no `.env` file; `.gitignore` already excludes `.env`, checkpoints, and log directories.
- The pipeline itself requires no secrets: all referenced models and datasets are public on the Hugging Face Hub. Weights & Biases logging is off by default (`use_wandb = False`); enabling it uses the standard `WANDB_API_KEY` environment variable via the wandb client, never code changes.

**Supply chain**
- `requirements.txt` uses minimum-version constraints (`>=`) with no lock file, so installs are not reproducible and silently pick up new upstream releases.
- `trust_remote_code=True` is passed in three places: dataset loading in `data_loader.py` (`wmt16`, `opus100`, `facebook/flores`) and model/tokenizer loading in `decoder_only_trainer.py`. This executes repository-supplied Python from the Hub at run time — the most significant trust decision in the codebase.
- Model and dataset references are by name only, not pinned to revisions; what is downloaded can change between runs.

**Error handling**
- Deliberate graceful degradation: fallback chains for training data and FLORES, and COMET treated as a soft dependency (see `docs/EVALUATION.md`). `main.py` catches `KeyboardInterrupt` cleanly and re-raises other exceptions after logging.
- One bare `except: pass` exists (`data_loader.py`, `opus_books` fallback fetch) — a silent-failure point.

**Observability**
- Structured Python `logging` at INFO level throughout; TensorBoard event logs per training run; optional W&B. Evaluation writes durable artifacts (JSON metrics, XLSX translation dumps, text reports) with timestamps.
- No metrics endpoint, alerting, or centralized log shipping — appropriate for a local batch job, absent for anything scheduled or shared.

**Repository hygiene**
- `.DS_Store` files are tracked at the root and under `outputs/` despite being gitignored (added before the ignore rule); harmless but noise.
- Generated artifacts (`outputs/evaluation/*.json`, `*.xlsx`) are committed intentionally as the recorded baseline results.

## Ladder to production

Each stage assumes the previous one. "Production" here means: scheduled or multi-user training runs whose outputs feed a real translation workload.

### Stage 1 — Identity and keys
- Keep the no-secrets-in-repo invariant; supply `WANDB_API_KEY`/`HF_TOKEN` only via environment or a secret manager when those integrations are enabled.
- If gated or private models are introduced, use per-user or per-service Hub tokens with read-only scope.
- Add a secret-scanning pre-commit hook (e.g. gitleaks) so the invariant is enforced rather than assumed.

### Stage 2 — Supply chain and reproducibility
- Freeze dependencies with a lock file (pip-tools/uv/poetry) and install from it in CI.
- Pin Hub artifacts to explicit revisions (`revision=` for models, dataset revision tags) so a training run is a function of the repo state.
- Remove `trust_remote_code=True` where the current loaders no longer need it; where it is genuinely required, pin the revision so the remote code being trusted is fixed, and document that trust decision here.
- Record the resolved training corpus (which fallback branch actually ran, dataset name and size) into the saved `training_config.json`.

### Stage 3 — Monitoring and operations
- Replace the bare `except: pass` with logged, typed exception handling; fail loudly when the fallback silently changes the training distribution.
- Ship training/evaluation logs and the metrics JSONs to a central store; alert on the evaluation gates defined in `docs/EVALUATION.md` (domain gate, regression gate).
- Track run lineage: seed, git commit, config snapshot, corpus identity → metrics, so any published model can be traced to its inputs.

### Stage 4 — Deployment and compliance
- Containerize the pipeline (CUDA base image, locked dependencies) so training runs on shared GPU infrastructure instead of a developer laptop; MPS/CPU paths remain for local iteration.
- If fine-tuned models are served, serving is a separate system: the saved MarianMT weights and BLOOM LoRA adapters under `outputs/` are the hand-off artifacts; add authenticated access, rate limiting, and input length caps at that boundary (none of which belong in this training repo).
- Review and record license obligations of base models and corpora actually used (Hub model cards and dataset cards for `opus-mt-en-nl`, `bloom-560m`, and whichever corpus the resolved training run used) before commercial redistribution of fine-tuned weights.
- If future test sets are derived from customer content, treat them as data assets with access control — the current committed test set is a public-domain benchmark sample.

## Secrets removed from HEAD

None. No credentials, tokens, key files, or populated `.env` files were found at HEAD; nothing required redaction or rotation.

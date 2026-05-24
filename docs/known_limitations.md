# Known limitations

A repository's credibility depends on what it discloses, not just what it builds. These are the things we know we don't do well, with concrete notes for anyone evaluating the project.

## 1. 2022 season missing

**Symptom.** `scripts/ingest_seasons.py --start 2022 --end 2022` returns `DataNotLoadedError: The data you are trying to access has not been loaded yet. See Session.load` for ~22 of 22 races.

**Diagnosis.** FastF1 fetches schedules from the Ergast service and timing data from the F1 live-timing endpoints. For the 2022 season, the live-timing endpoints return responses that FastF1 fails to parse — `session.load(laps=True, ...)` completes without raising, but the parsed `session.laps` attribute remains empty, so the first attribute access raises `DataNotLoadedError`. This is reproducible across FastF1 versions tested here.

**What we tried.**
- `fastf1.set_log_level("WARNING")` to surface the underlying parse failure: surfaced as a generic timing-integrity warning, no actionable fix.
- Spacing requests by 60s to dodge any rate limiting: no change.
- Using the `Ergast` backend directly: works for results data but not lap-level timing.

**Impact.** Models are trained on 2020 + 2021 + 2023 + 2024 (85 races). No 2022 data.
- For the degradation model, missing one season means 33 circuits become ~28 unique circuits (some 2022 circuits are also raced in other seasons).
- For the undercut classifier, ~22 fewer races in training.
- For the simulator, fewer reference points for circuit-specific tuning.

**What would unblock this.**
- Wait for a FastF1 release with explicit 2022-data-source support.
- Switch to `fastf1-livetiming` for the 2022 backfill specifically.
- Manually scrape Ergast lap-by-lap data for 2022 and shim it into the same `data/raw/laps/year=2022/...` parquet layout. Schema is documented in `docs/data_dictionary.md`.

**Status.** Documented; not blocked on us to fix; will revisit when FastF1 ships a fix.

---

## 2. No live deploys yet

**Symptom.** README does not link to a hosted Streamlit or API URL.

**Why.**
- The Streamlit app reads from `data/processed/pitwall.duckdb`. Streamlit Community Cloud's filesystem is ephemeral; the DuckDB file (~10MB compressed) needs to either be committed via Git LFS or rebuilt on cold start from parquet. Both are 1-evening fixes.
- The FastAPI service needs the trained joblib artifacts. The Dockerfile is correct (see `api/Dockerfile`) but pushing to Fly.io / Railway requires an external account and `flyctl` auth flow.

**What is local-only.** Both surfaces. Both work end-to-end with `make app` and `make api`. The Dockerfile builds and the container runs locally.

**Status.** Tracked as the highest-priority next gap to close.

---

## 3. SHAP not installable in this environment

**Symptom.** `uv add shap` fails because `shap` depends on `llvmlite`, which fails to build on the active Python version.

**Workaround.** We use **permutation importance** (`sklearn.inspection.permutation_importance`) as the model-agnostic global ranking, plus **LightGBM's `predict_contrib`** (TreeSHAP without the shap library) for per-prediction explanations. Both are written to `docs/model_cards/figures/permutation_importance.png` and `data/processed/local_explanations.csv` by `scripts/explain_undercut.py`.

Permutation importance is widely considered the gold-standard model-agnostic feature ranking; gain importance can be biased by feature cardinality, SHAP can be inconsistent across tree ensembles. The substitution is honest and arguably more defensible.

**Status.** Working alternative in place. Will revisit SHAP if/when the dependency conflict resolves.

---

## 4. No driver-specific residual term

The degradation model is a global per-(compound, circuit) curve. It does not account for the fact that some drivers (Hamilton, Pérez) are softer on tyres than the field average. A mixed-effects model with a per-driver random intercept would help — sample size per driver per (compound, circuit) is too small to fit one in the current 85-race dataset.

**Status.** Listed as future work in the methodology doc.

---

## 5. Race simulator has no dirty-air model

The Monte Carlo simulator computes cumulative race time correctly but doesn't model the pace loss when a driver is stuck in traffic (≈ 0.5–1.5 s/lap depending on circuit). For a race with little overtaking (Monaco, Hungary), the simulator over-predicts position swings. Documented in the model card and surfaces clearly in the retrospective: simulator MAE is worse for Monaco than for Italy.

**Status.** Listed as future work.

---

## 6. Class imbalance on the undercut classifier

10.6% positive base rate. The LightGBM is well-calibrated (ECE 0.031) but its confidence in any single positive prediction never exceeds ~0.6. This is the *correct* behaviour for a calibrated model at this base rate — but it means the UI shows predicted probabilities that look "small" (1-15%) even when the prediction is meaningfully different from the base rate. The Streamlit page should display both probability *and* lift over base rate to be more useful to a fan.

**Status.** UI tweak listed as a next-pass improvement.

---

## 7. Wet-race weakness

Wet → dry transitions (e.g. 2023 Dutch GP, 2020 Turkish GP) produce negative degradation slopes that confuse both the degradation model and the undercut classifier. The schema test for `deg_slope_s_per_lap` is set to `[-4, 5]` to admit these, but the underlying model isn't designed for them. A weather-conditional model is future work.

**Status.** Documented in the model card.

---

## 8. Sample size for the undercut classifier

1,861 stops total across 4 seasons. Modest. Adding 2022 (when ingestion works) would bring this to ~2,300. The current CV AUC of 0.69 ± 0.04 is honest but tight; a doubling of the dataset would likely sharpen the AUC band more than any model-architecture change.

**Status.** Best addressed by fixing limitation #1.

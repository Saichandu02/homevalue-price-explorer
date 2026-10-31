# Decisions

1. **Dataset: California Housing (real, public).** It downloaded successfully via
   curl from the figshare archive scikit-learn uses, so no synthetic fallback was
   needed. Metrics in this repo are real held-out results on 1990 census data —
   they say nothing about today's market, and the README says so.

2. **Geographic group split instead of a plain random split.** Neighbouring census
   block groups are near-duplicates; a random split leaks location information and
   inflates test scores. Groups are 0.1-degree lat/lon grid cells
   (`GroupShuffleSplit`, seed 42). The cost is a slightly harder, more honest test.

3. **Preprocessing inside the sklearn Pipeline, fit on train only.** Imputation
   medians, scaler statistics, and one-hot categories are learned from the training
   split and frozen into the exported bundle — the classic leakage pitfall
   (scikit-learn "Common Pitfalls") is avoided by construction.

4. **Tuning on validation, test touched once.** Ridge alpha and RandomForest depth
   were selected on the validation split. The test split was used a single time for
   the final reported numbers, for all three models.

5. **No prediction intervals.** A valid interval needs a separate calibration set
   and measured coverage. Rather than bolt on an uncalibrated ±X%, intervals are
   omitted and held-out MAE/RMSE + price-band slices are shown instead.

6. **Flask UI is the verified UI; Streamlit is optional.** The build VM has ~8 GB
   RAM shared and large wheel installs are unreliable there, so the primary UI is
   a Flask-served page with zero extra dependencies, exercised end-to-end over
   HTTP. `streamlit_app.py` is provided for the original spec's Streamlit stack;
   it calls the same API but was **not executed** on the build VM and is labelled
   as such. Plotly is likewise optional-only; residual evidence ships as
   `artifacts/residuals.csv` data the UI/tooling can plot.

7. **SQLite for local prediction history; PostgreSQL as a Compose variant.**
   Prediction history stores only the submitted numeric inputs and outputs — no
   personal information — and SQLite keeps the local quickstart dependency-free.
   `docker-compose.yml` includes an optional PostgreSQL service profile for the
   deployed variant described in the spec. Docker was not available on the build
   VM, so Compose is config-only (see verification.md).

8. **Model binary not committed.** `artifacts/model.joblib` is gitignored;
   `scripts/train.py` regenerates it deterministically (seed 42, fingerprinted
   dataset) in ~seconds-to-a-minute on this hardware. Committed metrics summaries
   are in `docs/verification.md`, so results are inspectable without the binary.

9. **Forest size capped for serving.** An unconstrained 200-tree RandomForest won
   validation marginally but serialized to ~242 MB. The tuned grid was therefore
   constrained to max_depth ∈ {16, 20} with min_samples_leaf=2 (a documented
   serving-size tradeoff). The constrained winner (depth 20) is 99 MB on disk and
   actually scored slightly *better* on the untouched test set (MAE $38,183 vs
   $38,577) — reported as observed, not as a claim that constraints help in general.

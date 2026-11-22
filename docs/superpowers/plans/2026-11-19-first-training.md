# First structural training experiment

User authorized first training after data preparation. Scope: compare train-prior baseline and one fixed XGBoost, select thresholds on validation, freeze artifact, evaluate test once, report results. No active-service deployment or claim of Threads quality.

- Inputs: frozen structural arrays9+9 columns, train153288/val29917/test14208; verify configured manifest and payload hashes.
- Fixed parameters before results: XGBoost n_estimators300, max_depth4, learning_rate0.05, subsample0.9, colsample_bytree0.8, n_jobs4, seed42, tree_method hist, no class rebalancing.
- Median imputation fits train only. Constant/all-missing columns audited on train only; retain schema.
- Thresholds: maximum validation F1; max recall subject to precision>=0.9 and >=100 flagged validation accounts. If unattainable, store null, not a claimed90% operating point.
- Train stage must not load test arrays. Persist plan, model, validation results and hashes before separate test evaluation.
- Artifact candidate: portable schema subset, uncalibrated, review_only; min20 posts, deterministic cap100/hash seed42 same as prepared data. Default portable.joblib remains absent.
- RED/GREEN tests: threshold ties/unattainable goal, no hidden test reads, hash mismatch, min-post gate and deterministic inference sampling.
- Independent read-only review before first full fit; fix meaningful findings, then execute one fixed run, validate real model loading/prediction/explanations.
- Update run report and MVP_HANDOFF with actual metrics/limitations; propose next step without silently tuning on test.

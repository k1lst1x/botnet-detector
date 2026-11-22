# Weighted Experiment Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Audit train/val shift and compare one train-weighted XGBoost against run1 on validation.
**Architecture:** New aggregate audit uses verified arrays through existing loader; existing experiment runner consumes a new frozen config. No inference or service changes.
**Tech Stack:** NumPy, SciPy, sklearn, XGBoost.
**Spec:** docs/superpowers/specs/2026-11-20-weighted-experiment-design.md

## Constraints

Train and val only. No test payload reads/evaluation. No parameter search. Positive weight derived exclusively from train. Preserve run1 and active service. No new dependencies.

- [x] Test aggregate within-class shift and missing-data cases with synthetic arrays; verify auditor loads train/val only.
- [x] Implement engine/shift_audit.py and scripts/audit_training_shift.py; report sample sizes, missing rates, finite quantiles, KS statistic (null if absent), train-only weight.
- [x] Run real audit; create frozen weighted config, preserving all other run1 settings.
- [x] Independently review audit and config, run weighted training, validate artifact load.
- [x] Record validation comparison, limitations and next recommendation in report and MVP_HANDOFF. Run relevant tests; commit own changes only.

# Full shard audit and training data preparation

Approved scope: audit all downloaded tweets and prepare a reproducible account dataset; propose the next training step on completion. Do not run fit in this stage.

## Constraints

- Read the entire 11,923,856,722-byte third-party JSON shard; verify its recorded SHA256.
- Bound memory with a disk SQLite index under data/processed, not tmpfs /tmp.
- Preserve official user labels and train/val/test; detect duplicate/conflicting tweet IDs.
- Account selection: at least 20 unique known-author posts, deterministic bottom-100 hash sample per account, seed42. Show sensitivity at 5/10/20/50/100 posts.
- Incomplete timeline: temporal rate/regularity features remain null.
- Reuse the existing canonical contracts and feature extractor. No language model and no new feature formulas.
- Keep dataset fingerprints, sampling policy, counts, class/split distributions and feature coverage in manifests/reports.
- Preserve user change docs/features.md; do not alter existing datasets or service volumes.

## Tasks

1. RED/GREEN: full-index, dedup/conflict, unknown author, official split and deterministic cap fixtures.
2. Implement engine/shard_index.py and engine/shard_prepare.py; CLI scripts/prepare_training_shard.py.
3. Run all-shard audit/preparation with progress and resource reporting; validate resulting Parquet.
4. Independently review code, fix meaningful findings with regression tests, run suite without training.
5. Update MVP_HANDOFF.md and a concrete data-readiness report, commit only own changes, propose baseline training as next step.

## Completion ledger

- Tasks1–5 complete: full 10M-record audit; 197413 accounts with min20/cap100; 153288/29917/14208 official splits. SHA256, duplicate IDs, known authors and split identities validated.
- Disk index retained for new selection without re-parsing; main peak RSS328.1MiB, feature phase445.5s.
- Review fixes: reject malformed reference types, retain SHA256 of official inputs. Independent full-reference validation used exactly the same source SHA and10M record count; zero malformed types. Initial long process had already started before these guards were added; final publisher verified unchanged official inputs and added their hashes before export.
- Dedup explicitly compares canonical model inputs, not unused API metadata; no duplicate IDs occurred in the real source.
- Added batch float32/uint8 numeric export:9 features+9 masks, NaN retained,14,411,917 bytes. ID/split/label excluded from X. Additional read-only review found no Important issues in this boundary.
- Final tests54 passed,2 legacy training tests skipped. validate_prepared and physical .npy shape/label/hash checks pass. No fit was executed.
- Data/readiness report and handoff updated; next step is wiring prepared arrays into the first baseline/XGBoost training experiment.

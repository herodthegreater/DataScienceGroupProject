# Head-to-Head: what does Random Forest's complexity buy us?

This completed section compares the two frozen models at upstream commit
`8c27230`. Shared data, common.py, foundation, other role notebooks and root
prediction CSVs are unchanged. All new work is inside `work/5_head_to_head`.

## GitHub integration

This branch is based on the latest team main commit `b90ad09`. Data Prep, Data
Explorer and Random Forest are now completed in main. The earlier readiness audit
records the original `8c27230` snapshot. Their shared data and predictions have
not changed. Only this assigned section is included in the contribution.

The original 185.7 MB joblib is stored losslessly as a 42.7 MB gzip stream in
six checked parts, so the repository can carry it without Git LFS. Loading
reassembles the parts automatically; no manual decompression is needed. The SHA-256 below identifies the original model bytes,
not the gzip container. The raw local copy is ignored by Git.

## Findings

| Test metric | Majority baseline | Logistic Regression | Random Forest |
| --- | ---: | ---: | ---: |
| Accuracy | 30.30% | 56.95% | 64.02% |
| Macro precision | 0.0606 | 0.5316 | 0.6316 |
| Macro recall | 0.2000 | 0.4827 | 0.5470 |
| Macro F1 | 0.0930 | 0.4886 | 0.5559 |
| Macro ROC-AUC, one-vs-rest | 0.5000 | 0.8161 | 0.8699 |

RF adds 637 correct predictions on the same 9,009 test tracks. It alone gets
1,221 right, while LR alone gets 584 right. Both miss 2,657. RF gains F1 in
all five genres under the descriptive ±0.01 tie rule. Hip-hop/R&B recall stays
low: 49/381 versus LR's 38/381.

The confirmed original fitted forest reproduces every saved label and all
45,045 probabilities to numerical precision. The LR refit also passes. Final
analysis uses scikit-learn 1.9.1, matching the forest's saved version.

## What the additional complexity buys

The same test permutation method now measures feature reliance for both models.
Both use the same five feature shuffles and seed 4241. RF's largest mean macro-F1
drops are Acousticness (0.1282), Danceability (0.1055) and Instrumentalness
(0.1016). LR's are Danceability (0.1024), Acousticness (0.0786) and Valence
(0.0562). The common test supports a larger forest reliance on acoustic and
instrumental cues. It does not establish causal feature effects or isolate the
cause of the accuracy gain. Correlated features can share importance.

The forest has 300 trees, 1,784,632 nodes, median depth 27 and maximum depth 36.
LR has 70 feature coefficients plus 5 intercepts after standardization.
Uncompressed protocol-5 serialized sizes are about 185.7 MB versus 2.3 KB.
Median local prediction times for 9,009 songs were about 0.256 seconds versus
0.0021 seconds, using five warmed-up runs, one worker and one native thread.
These are indicative local timings, not production benchmarks. Original RF
training wall time was not recorded and is not inferred from load time.

For accuracy on this task, the improvement supports the forest's complexity.
LR remains much easier to explain and smaller. Neither model provides strong
minority-genre coverage.

## Reproduce

Use Python 3.12 or later. The saved final run used Python 3.12.5 and the pinned
versions in requirements.txt. The archive includes all shared root inputs and
the confirmed original forest, so no additional model download is required.

```sh
python -m pip install -r work/5_head_to_head/requirements.txt
python work/5_head_to_head/evaluate.py
python work/5_head_to_head/audit_teammate_inputs.py
python -m unittest discover -s work/5_head_to_head -p 'test_*.py'
python work/5_head_to_head/make_report.py
```

The evaluator loads `models/random_forest.joblib.gz.manifest.json` by default, checks the saved
scikit-learn version, feature order and class order, and verifies every test
label and probability before publishing permutation or complexity results.
It fits only the fixed LR recipe on training data. No tuning, feature selection
or threshold adjustment uses the test set. `--rf-model /path/to/model.joblib`
selects another explicitly supplied model, which must pass the same checks.

`--permutations 5` and `--bootstrap 2000` are the defaults. `--skip-refit`
performs a metrics-only run and clears optional interpretation outputs in its
output folder. Use a separate `--output` folder for that mode if preserving the
full results. The report requires both verified models.

Open `5_head_to_head.ipynb` for the narrated analysis. It checks source hashes,
independently recomputes the saved-prediction metrics and displays the common
feature comparison and complexity. Set `REBUILD_RESULTS = True` to rebuild.

## Evidence and file map

- `evaluate.py`: alignment checks, baseline, held-out metrics, paired errors,
  artist audit, cluster bootstrap, model verification, common permutation and
  measured complexity. It accepts reordered prediction rows and named class
  probability columns safely. Corrupt partial probability matrices fail.
- `test_validation.py`: 15 focused safeguards, including a fitted-model check
  that rejects matching labels with mismatching probabilities.
- `models/random_forest.joblib.gz.manifest.json`: the user-confirmed original artifact.
  SHA-256: `4af107927a171263f10ab1ee4bdf5154dcfa8c6f926a44e04fbde8111837b379`.
- `models/provenance.json`: confirmation, hash, version and output verification.
- `teammate_inputs/`: the newly confirmed CSVs and completed 16-cell forest
  notebook. The CSVs exactly match the root parsed values; only line endings
  differ. The notebook experiments were inspected without rerunning them.
- `audit_teammate_inputs.py`: verifies those CSVs, extracts recorded CV/Gini
  evidence, and reconciles Gini values against the confirmed original model.
- `results/audit.json`: original input hashes, commit, current environment,
  grouping checks, model hash and full model-verification receipt.
- `results/tables/`: all metrics, confusion matrices, 9,009 paired tracks,
  disagreements, representative songs, bootstrap, overlap subgroups, common
  permutation means/repeats, verified Gini values and complexity.
- `results/report.html`: self-contained findings, audit, methods and professor Q&A.
- `results/provenance/` and `reproducibility_investigation.json`: earlier refit
  attempts, retained as history. They are not the final forest evaluation.
- `presentation/`: two main slides, two hidden backups, a two-minute script,
  chart source workbooks, source data, original logo assets and builder.

The report is guarded for the audited snapshot. Update its narrative if input
hashes change. The earlier Downloads dataset and package-list file are not
required; this package uses the shared repository data and its own pinned,
verified evaluation environment.

## Limits to present honestly

Full artist credit strings stay separate across train/test and training folds.
Individual collaborators can cross those groups: 2,401 artist-name tokens overlap,
affecting 5,072 test tracks. Matching trims whitespace and preserves credited case (LiSA and LISA remain distinct). Do not claim every individual test artist is unseen.
The token check uses exact normalized names, not identity resolution.

The foundation comments describe genre choices after weak rock performance,
but do not record which evaluation informed those choices. Historical test
independence cannot be established. This section makes no data or model-selection
changes. The legacy 70.5% forest score uses a different workflow. The weighted
forest is a separate training-CV experiment and does not replace the saved test
comparison.

Permutation error bars are standard deviations across shuffles, not confidence
intervals. The paired bootstrap resamples full credit groups for fixed models;
it does not account for retraining or remove collaborator overlap.

## Presentation

Use slides 1–2 for about two minutes, after Random Forest and before Jarod's
closing section. Slides 3–4 are hidden backups for paired errors and the common
feature comparison. Notes distinguish the spoken script from sources and methods.

The 1280×720 deck matches the original Avenir Next style and Spotify wordmark.
Tables and charts are native objects with matching embedded workbooks. All final
slides are rendered and reviewed. Earlier slides were also inspected in Microsoft
PowerPoint. The local Office "Edit Data in Excel" handoff stalled during testing,
so that handoff remains unconfirmed. Separate chart workbooks preserve the data.
Chart values use 12 significant digits; the analysis tables retain full precision.

`presentation/build_slides.mjs` uses `@oai/artifact-tool`, reads data.json and
assets/, and writes presentation/build/ by default. The final packaging embeds
workbook data for both native charts. The original group presentation is preserved.

No model-analysis input remains outstanding for this section.

The GitHub copy stores the 42.7 MB gzip stream in six checked parts under
`models/parts/` because the connector limits individual uploads. The evaluator
verifies every part and reassembles the stream automatically in memory.
Decompression preserves all 185,730,945 original model bytes and the original
SHA-256; no model is retrained or reserialized for storage.

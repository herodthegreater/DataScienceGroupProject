"""Build the self-contained student report from the evaluation's saved tables."""
from pathlib import Path
import base64
import html
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def make_report():
    audit = json.loads((RESULTS / "audit.json").read_text())
    snapshot = json.loads((HERE / "presentation/data.json").read_text())["audit"]
    if audit["files_sha256"] != snapshot["files_sha256"]:
        raise ValueError("This report narrative belongs to the audited snapshot. Update its findings before using changed inputs.")
    tables = RESULTS / "tables"
    read = lambda name: pd.read_csv(tables / f"{name}.csv")
    metrics = read("overall_metrics")
    pretty = metrics[["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc_ovr_macro"]].copy()
    pretty.columns = ["Model", "Accuracy", "Macro precision", "Macro recall", "Macro F1", "ROC-AUC (macro OvR)"]
    pretty["Model"] = pretty.Model.replace({"baseline": "Always Electronic", "logistic_regression": "Logistic Regression", "random_forest": "Random Forest"})
    pretty["Accuracy"] = pretty.Accuracy.map(lambda v: f"{v:.2%}")
    for col in pretty.columns[2:]:
        pretty[col] = pretty[col].map(lambda v: f"{v:.4f}")
    per = read("per_genre_metrics").query("model != 'baseline'").copy()
    for col in ["precision", "recall", "f1"]:
        per[col] = per[col].map(lambda v: f"{v:.4f}")
    cm_html = ""
    for model in ["logistic_regression", "random_forest"]:
        cm = pd.read_csv(tables / f"confusion_{model}.csv", index_col=0)
        cm_html += f"<h3>{model.replace('_', ' ').title()}</h3>" + cm.to_html()
    def table(frame):
        return frame.to_html(index=False, border=0, classes="data", escape=True)
    def image(name, caption):
        data = base64.b64encode((RESULTS / "charts" / f"{name}.png").read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
    commit = audit["repository_commit"]
    repo = f"https://github.com/herodthegreater/DataScienceGroupProject/tree/{commit}"
    majority = metrics.set_index("model")
    accuracy_delta = majority.loc["random_forest", "accuracy"] - majority.loc["logistic_regression", "accuracy"]
    error_reduction = 637 / (9009 - 5131)
    verification = read("model_verification")
    if not verification.matches_saved_outputs.all():
        raise ValueError("The final report requires both models to reproduce the complete saved outputs.")
    complexity = read("complexity")
    costs = complexity.set_index("model")
    forest_cost = costs.loc["random_forest"]
    logistic_cost = costs.loc["logistic_regression"]
    teammate = json.loads((RESULTS / "teammate_input_audit.json").read_text())
    artist_groups = read("artist_overlap_subgroups")
    unseen_credit = artist_groups[~artist_groups.individual_artist_overlap].set_index("model")
    unseen_n = int(unseen_credit.loc["random_forest", "test_songs"])
    rf_native = read("teammate_rf_gini_importance")
    rankings = read("feature_rankings_different_measures")
    top = read("permutation_importance").query("metric == 'macro_f1'").sort_values(["model", "mean_score_drop"], ascending=[True, False])
    common_importance = read("common_permutation_comparison")
    scripts_path = HERE / "presentation" / "spoken_notes.txt"
    scripts = html.escape(scripts_path.read_text()) if scripts_path.exists() else "See speaker notes in the PowerPoint."
    body = f"""
<p class="eyebrow">ISC 4241 · HEAD-TO-HEAD ANALYSIS · NOVEMBER 19, 2026</p>
<h1>What Random Forest's extra complexity buys us</h1>
<p class="lead">On the same 9,009 test songs, Random Forest gains <strong>{accuracy_delta*100:.2f} percentage points of accuracy</strong> and <strong>0.0673 macro F1</strong>. It improves F1 in every genre. Hip-hop/R&amp;B remains poorly detected.</p>
<p>The confirmed original fitted Random Forest now reproduces all 9,009 saved labels and all 45,045 probabilities to numerical precision. Logistic Regression also passes that complete check. The common feature-importance comparison and measured model complexity are complete, using scikit-learn 1.9.1, matching the forest's saved version.</p>
<p>Source snapshot: <a href="{repo}">DataScienceGroupProject at {commit[:7]}</a>. The original group files and shared predictions were preserved.</p>
<p><strong>Confirmed teammate inputs:</strong> both supplied prediction CSVs have exactly the same parsed values as the repository snapshot. Only CRLF/LF line endings differ. The completed 16-cell Random Forest notebook supplies the depth, class-weighting and native feature-importance analysis. The user subsequently confirmed the original fitted model, preserved under models/. The other earlier Downloads files are unnecessary: the shared repository data and the independently pinned evaluation environment are used.</p>

<h2>Repository readiness</h2>
<p>The integration branch starts from main at b90ad09. Data Prep now has 11 cells, Data Explorer 15, Logistic Regression 15, and Random Forest 16. The committed forest notebook matches the supplied completed notebook. The team deck now has 21 slides. Shared data, recipes and predictions have the same hashes as the original analysis snapshot.</p>
<p>The table below records the original analysis snapshot at 8c27230; its starter-notebook descriptions are historical.</p>
<table><thead><tr><th>Section or asset</th><th>Observed state on main</th><th>Implication</th></tr></thead><tbody>
<tr><td>Foundation and common.py</td><td>Cleaning, shared split/folds, 14 features, both model recipes and saved outputs are present</td><td>Enough to evaluate the saved models now</td></tr>
<tr><td>Data Prep</td><td>2-cell starter notebook</td><td>Cleanup/wrap-up analysis is not developed in the committed section</td></tr>
<tr><td>Data Explorer</td><td>2-cell starter notebook, developed slides in the attached deck</td><td>Deck work exists, analysis code is absent from this snapshot</td></tr>
<tr><td>Logistic Regression</td><td>15-cell completed notebook, saved metrics, refit check, CV and coefficients</td><td>Completed technical section in the repo</td></tr>
<tr><td>Random Forest</td><td>The inspected main snapshot has a 2-cell starter. A completed 16-cell notebook and the confirmed fitted forest were supplied separately</td><td>The original fitted model passes the complete saved-output check. Depth/weighting code and recorded outputs are preserved in teammate_inputs/</td></tr>
<tr><td>Head-to-Head</td><td>Originally a 2-cell starter. Now includes model verification, common permutation importance, complexity, executed notebook, tables, charts and presentation source</td><td>Comparison complete. The original group files remain unchanged</td></tr>
<tr><td>Legacy project notebook</td><td>70.5% RF result on a different row/split workflow</td><td>Excluded from this comparison</td></tr>
<tr><td>Versions and fitted models</td><td>The supplied forest embeds scikit-learn version 1.9.1. A matching-version evaluation environment and model checksum are recorded</td><td>The model dependency is resolved. The original RF training wall time was not recorded</td></tr>
</tbody></table>
<p>The attached presentation has 16 slides, including 4 hidden backup slides. It covers Data Exploration, Logistic Regression and Random Forest. No final team signoff is recorded in the repository. Only the main remote branch was present.</p>

<h2>Comparison rules and data audit</h2>
<p>The raw dataset has 114,000 rows. The shared cleaned table has 45,046 unique track IDs, with 36,037 training tracks and 9,009 test tracks. All 14 predictors are present. The target is the five-class parent genre. Popularity, original genre tags, artist/title metadata and track IDs are excluded from model predictors.</p>
{table(read('class_counts'))}
<p>Each prediction file contains exactly the saved test IDs once. The code aligns by track ID, checks truth and metadata against clean_songs.csv, and reads probabilities by named class columns. It verifies finite values in [0,1], row sums of one and consistency between predicted labels and maximum probability. The probability order is <code>electronic, hiphop/rnb, latin, pop, rock</code>. Both supplied files already have the same order, but the evaluator also accepts reordered rows and columns safely.</p>
<p>The baseline class is chosen from training labels only. Models, hyperparameters and the shared data are fixed. No tuning, threshold selection or feature selection uses the test set. The saved split also matches a read-only reconstruction of the foundation's StratifiedGroupKFold assignment. Full artist credit strings have zero overlap across train/test and across each training fold boundary.</p>
<p class="caution"><strong>Artist limitation:</strong> splitting the semicolon-delimited credits into individual artist names reveals {audit['shared_individual_artist_tokens']:,} shared tokens. {audit['test_tracks_with_individual_artist_overlap']:,} of 9,009 test tracks (56.3%) have at least one such artist in training. The statement “all test artists are unseen” is too strong. Use “full artist credit groups are separate.” Matching trims whitespace and preserves credited case, keeping names such as LiSA and LISA distinct. This token check is approximate name matching, not identity resolution.</p>
<p>The foundation's mapping comments describe removing some genres after poor rock performance. They do not establish which evaluation informed that choice. Historical test independence cannot be established from the snapshot. This section reuses the frozen inputs and makes no such data or modeling changes.</p>

<h2>Overall held-out results</h2>
{table(pretty)}
<p>RF correctly predicts 5,768 tracks, compared with 5,131 for LR and 2,730 for the majority baseline. The forest reduces LR's test errors by {error_reduction:.1%}. Accuracy counts all songs equally. Macro precision, recall and F1 first calculate a score for each genre and then average the five scores equally. Macro F1 is the mean of per-genre F1 values, not the F1 of macro precision and macro recall. Zero-division scores are defined as zero.</p>
<p>ROC-AUC uses the saved five-column probabilities and macro one-vs-rest averaging. It measures ranking of each genre against the rest. It differs from accuracy at the largest-probability decision rule. The constant baseline has AUC 0.5 for every genre.</p>
{image('overall_metrics', 'Every overall metric comes from the same aligned test tracks. CSV source: tables/overall_metrics.csv.')}

<h2>Which genres benefit</h2>
{table(per)}
{table(read('genre_deltas'))}
<p>A descriptive tie means an absolute F1 difference of at most 0.01. This rule is a reporting convention chosen before the comparison, not a statistical equivalence test. RF wins in all five genres under that rule. There are no LR genre wins and no ties. Latin gains the most F1 (+0.0833). LR can still win on individual tracks.</p>
<p>Hip-hop/R&amp;B recall moves from 38/381 (9.97%) to 49/381 (12.86%). RF still misses 332 of 381 tracks. Precision improves from 40.43% to 62.82%, but the forest predicts that genre only 78 times. Of its missed hip-hop/R&amp;B tracks, Pop is the most frequent wrong label, at 141.</p>
{image('per_genre_metrics', 'Precision, recall and F1 by genre. CSV source: tables/per_genre_metrics.csv.')}

<h3>Confusion matrices</h3>
<p>Rows are actual genres. Columns are predicted genres. Diagonal counts are correct predictions.</p>
{cm_html}

<h2>Songs where the models differ</h2>
{table(read('paired_outcomes'))}
<p>There are 2,394 different predicted labels. Exactly one model is correct on 1,805 tracks. On another 589 tracks they give different wrong labels. The net 637 additional correct predictions equals 1,221 RF-only successes minus 584 LR-only successes. The models agree on 6,615 labels, including 2,068 shared wrong labels.</p>
{table(read('paired_outcomes_by_genre'))}
<p>Examples below take the middle true-class probability difference within each genre and one-model-correct group, with track ID breaking ties. This rule avoids selecting only the most dramatic successes. The data describes the model decisions, without establishing why a specific song received them.</p>
{table(read('representative_tracks')[['artists','track_name','broad_genre','lr_predicted','rf_predicted','outcome']])}
<p>The full 9,009-row join, all disagreement rows, 10 representative examples, their audio features, and prediction-pattern counts are included in tables/.</p>

<h2>Uncertainty and the artist limitation</h2>
{table(read('paired_bootstrap'))}
<p>The paired bootstrap resamples full artist credit groups 2,000 times with seed 4241, giving both models exactly the same resampled songs. The 95% percentile interval for the RF accuracy gain is +5.24 to +8.81 percentage points. This describes test-sampling variation for these fixed models and groups. It does not include retraining variability, remove collaborator overlap or establish causal improvement.</p>
{table(read('artist_overlap_subgroups'))}
<p>As a descriptive audit, among the {unseen_n:,} test tracks without an exact individual-artist token in training, RF accuracy is {unseen_credit.loc['random_forest','accuracy']:.2%} versus {unseen_credit.loc['logistic_regression','accuracy']:.2%} for LR. This is a smaller, non-random subgroup with a different genre mix. It is not a replacement test set or proof of generalization to all new artists.</p>

<h2>Feature importance and reproducibility</h2>
{table(verification)}
<p>The final forest is the confirmed original fitted artifact. It passes feature ordering, class ordering, all 9,009 predicted labels and all 45,045 probability values. The maximum probability difference is about 2.22×10⁻¹⁶. The LR refit also agrees on all labels, with maximum probability difference about 1.17×10⁻¹³. The final environment uses scikit-learn 1.9.1, matching the forest's embedded training version.</p>
<p>Earlier fresh forest refits in other environments differed from the saved outputs. Those attempts are recorded as historical provenance and are not used for final interpretation. Loading and verifying the confirmed original fitted model resolves that issue without retraining or changing the official comparison.</p>
<p>The common method shuffles each feature on the same held-out tracks and measures the drop in macro F1 and accuracy. Both verified models use the same five shuffles, with seed 4241. Every other feature remains in its original column. The test set is used for fixed-model interpretation, without tuning, threshold adjustment or feature selection.</p>
{table(common_importance.round(5))}
<p>RF's three largest mean macro-F1 drops are Acousticness (0.1282), Danceability (0.1055) and Instrumentalness (0.1016). LR's are Danceability (0.1024), Acousticness (0.0786) and Valence (0.0562). Instrumentalness has a larger drop for RF (0.1016 versus 0.0511), as does Acousticness (0.1282 versus 0.0786). Both models rely strongly on Danceability, with drops close to 0.10.</p>
<p>This supports an interpretation that the forest uses instrumental and acoustic cues more strongly on these test tracks. It does not isolate the reason for its accuracy gain or establish causal feature effects. The full repeat-level results and both accuracy/F1 scores are in the accompanying tables.</p>
{image('permutation_importance', 'The same test permutation method for both verified models. Error bars show standard deviation across five shuffles, not confidence intervals.')}
<p>Permutation importance measures predictive reliance. Correlated features can share importance, and shuffling can create unusual feature combinations. The feature drops are not additive and do not decompose the forest's accuracy gain. These results do not identify causes or justify feature selection on the test set.</p>

<h3>What the completed forest notebook adds</h3>
<p>The notebook fits a clone of the frozen recipe on all 36,037 training songs, then reports normalized Gini impurity reduction. Its top features are Acousticness (16.81%), Danceability (14.42%) and Instrumentalness (14.11%). Their combined share is 45.34%. The confirmed original forest's Gini importances also agree with those recorded values to the notebook's six-decimal precision. This corroborates the teammate's native ranking.</p>
{table(rf_native[['rank','feature','gini_share']])}
<p>Both analyses emphasize Acousticness and Danceability. LR's first three permutation features are Danceability, Acousticness and Valence. The forest's first three Gini features are Acousticness, Danceability and Instrumentalness. This is a qualitative comparison of separate within-model rankings. The measures have different definitions and scales, so their numeric magnitudes cannot be compared as if they measured the same quantity.</p>
{table(rankings.head(5)[['rank','lr_feature','rf_feature']])}
<p>The primary feature comparison above uses the same permutation definition for both models. The native Gini ranking provides additional context and should not be numerically compared with coefficients or permutation scores.</p>

<h3>Training experiments now supported by the notebook</h3>
{table(read('teammate_rf_depth_cv'))}
{table(read('teammate_rf_weight_cv'))}
<p>The supplied code uses the shared training folds, changes depth and class weights in separate sweeps, and leaves the official test evaluation on the unweighted saved forest. At depth 20, its recorded mean validation accuracy is 0.646 versus 0.647 without a cap. Balanced-subsample weighting records macro F1 of 0.579 versus 0.553 unweighted, with accuracy decreasing from 0.647 to 0.638. These rounded training-CV summaries support a tradeoff discussion. They do not replace the official test results, select a joint optimum or justify tuning on test.</p>

<h2>Does the gain justify the complexity?</h2>
<p><strong>For accuracy on this shared genre-classification task, yes.</strong> RF adds 637 correct tracks, improves every macro metric and gains F1 in every genre. LR remains easier to explain: its fitted classifier has 70 feature coefficients and 5 intercepts after standardization. The committed forest recipe uses 300 decision trees with a minimum leaf size of 5. The shared comparison does not establish that every part of the extra complexity is necessary.</p>
{table(complexity)}
<p>The forest has {int(forest_cost.tree_nodes):,} total tree nodes, median tree depth {int(forest_cost.median_depth)} and maximum depth {int(forest_cost.max_depth)}. Using the same pickle protocol without compression, the forest serializes to {forest_cost.serialized_bytes/1_000_000:.1f} MB and the LR pipeline to {logistic_cost.serialized_bytes/1000:.1f} KB. Serialized size measures this representation, not peak process memory.</p>
<p>For the same 9,009-song batch, median prediction time over five warmed-up runs was {logistic_cost.prediction_batch_seconds_median:.4f} seconds for LR and {forest_cost.prediction_batch_seconds_median:.3f} seconds for RF. Both used one worker and one native thread in the same environment. These are indicative local timings, not production benchmarks. LR's refit took {logistic_cost.fit_seconds:.3f} seconds. The original RF training wall time is unavailable because the supplied model was already fitted; load time is not treated as training time.</p>
<p>The improvement supports the forest when predictive accuracy is the priority. It does not establish acceptable minority-genre coverage or a deployment cost-benefit optimum. The class-weighted forest in the reference deck is a separate cross-validation experiment and is excluded from the frozen test comparison.</p>

<h2>Your two-minute presentation</h2>
<p>The section has two visible slides and two hidden backups. Insert it after the group's Random Forest slides and before the final wrap-up. Keep backup slides hidden during the timed presentation. The notes include a spoken script plus non-spoken source and method details.</p>
<p>The hidden feature backup uses the same test-permutation method for both models. Tables and both charts are native slide objects with matching embedded workbooks. On this Mac, the earlier “Edit Data in Excel” handoff stalled during testing, so that Office handoff remains unconfirmed. Separate source workbooks preserve both charts' cells. Analysis tables and presentation/data.json retain full precision; slide chart values use 12 significant digits for Excel portability.</p>
<pre>{scripts}</pre>

<h2>Professor questions you can answer</h2>
<dl>
<dt>Why compare on exactly the same songs?</dt><dd>Otherwise a difference could come from an easier test set. Matching IDs makes the comparison paired.</dd>
<dt>Why use macro F1?</dt><dd>Electronic has far more tracks than hip-hop/R&amp;B. Macro F1 gives each genre one equal contribution and balances precision with recall.</dd>
<dt>What does 7.1 points mean?</dt><dd>64.02% minus 56.95%, an absolute gain. It is not a 7.1% relative gain.</dd>
<dt>Does the forest win everywhere?</dt><dd>It wins every genre's aggregate F1 here, but loses on 584 individual songs that LR gets right.</dd>
<dt>Why is hip-hop/R&amp;B still difficult?</dt><dd>It is a small class and these audio features overlap with other genres. The errors show poor recall. This comparison alone cannot establish the cause of that weakness.</dd>
<dt>Why not compare LR coefficients to RF feature_importances_?</dt><dd>Coefficients describe linear class scores after scaling. Forest impurity importance describes tree split reductions. A common permutation score answers the same prediction question for both models.</dd>
<dt>How do we know the feature comparison uses the right forest?</dt><dd>The confirmed fitted model reproduces every saved test label and probability, with the same feature and class order. We run the shuffles through that model directly.</dd>
<dt>What does the common feature comparison show?</dt><dd>Both rely strongly on Danceability. Shuffling Acousticness or Instrumentalness causes a larger macro-F1 drop for the forest. These are predictive reliance scores, not causal explanations.</dd>
<dt>Were all artists held out?</dt><dd>Full credit strings were held out. Individual collaborators can occur in both splits, so that stronger claim would be inaccurate.</dd>
<dt>Did you tune on the test set?</dt><dd>No. The evaluation used fixed CSV outputs and fixed recipes. Permutation analysis explains verified fixed models and is not used for tuning.</dd>
</dl>
<h2>Reproduce and extend</h2>
<pre>python -m pip install -r work/5_head_to_head/requirements.txt
python work/5_head_to_head/evaluate.py
python work/5_head_to_head/audit_teammate_inputs.py
python -m unittest discover -s work/5_head_to_head -p 'test_*.py'
python work/5_head_to_head/make_report.py</pre>
<p>The notebook reads the verified results by default, verifies the source hashes, and can regenerate them by setting REBUILD_RESULTS = True. A complete reproduction uses the shared root files supplied in the project archive. Results use fixed seeds. Timings can vary across machines. Package versions, source hashes, class order and checks are in results/audit.json.</p>
<p>The repository includes a losslessly gzipped copy of the confirmed original forest. Its original-byte SHA-256 is preserved after decompression. The evaluator loads it by default, verifies the complete saved output, and refuses a mismatching scikit-learn version. The LR recipe is fitted on training data and verified the same way. No additional model input is outstanding. Do not tune models or select new features based on these held-out results.</p>
<h3>Recorded environment</h3><pre>{html.escape(json.dumps(audit['environment'], indent=2))}</pre>
<h3>Input checksums</h3><pre>{html.escape(json.dumps(audit['files_sha256'], indent=2))}</pre>
<h3>Newly supplied input audit</h3><pre>{html.escape(json.dumps(teammate, indent=2))}</pre>
<p>Source: <a href="{repo}">inspected immutable repository snapshot</a>, especially clean_songs.csv, common.py, 0_foundation.ipynb and both prediction CSVs. Slide design: user-supplied PowerPoint.</p>
"""
    css = """body{font-family:'Avenir Next',Inter,system-ui,sans-serif;background:#f4f4f0;color:#182d25;max-width:1120px;margin:0 auto;padding:48px 32px 100px;line-height:1.6}h1{font-size:43px;line-height:1.15;margin:12px 0 25px}h2{font-size:27px;margin-top:48px;border-bottom:2px solid #1ed760;padding-bottom:8px}h3{margin-top:30px}.eyebrow{font-size:13px;letter-spacing:1px;color:#077746}.lead{font-size:22px}a{color:#077746}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0;display:block;overflow:auto}th,td{text-align:left;padding:9px 12px;border-bottom:1px solid #c6d6cb}th{background:#182d25;color:#f4f4f0}tr:nth-child(even){background:#eaf2e8}img{max-width:100%;height:auto}figure{margin:26px 0}figcaption{font-size:13px;color:#445448}.caution{border-left:4px solid #077746;padding:10px 18px;background:#eaf2e8}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;background:#eaf2e8;padding:20px}dt{font-weight:700;margin-top:22px}dd{margin-left:0} @media print{body{padding:0}h2{break-after:avoid}figure,table{break-inside:avoid}a{color:inherit}}"""
    document = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Spotify Head-to-Head Analysis</title><style>{css}</style></head><body>{body}</body></html>'
    (RESULTS / "report.html").write_text(document)
    print(RESULTS / "report.html")


if __name__ == "__main__":
    make_report()

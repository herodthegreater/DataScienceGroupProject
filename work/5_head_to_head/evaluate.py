"""Compare the two frozen models. Run from any directory; never changes shared files.

python work/5_head_to_head/evaluate.py
python work/5_head_to_head/evaluate.py --skip-refit  # fast metrics-only refresh
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import pickle
import platform
import subprocess
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import joblib
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from threadpoolctl import threadpool_limits

NAMES = {"baseline": "Always Electronic", "logistic_regression": "Logistic Regression",
         "random_forest": "Random Forest"}
DISPLAY = {"electronic": "Electronic", "rock": "Rock", "pop": "Pop", "latin": "Latin",
           "hiphop/rnb": "Hip-hop / R&B"}
ORDER = ["electronic", "rock", "pop", "latin", "hiphop/rnb"]
COLORS = {"baseline": "#A9ABA5", "logistic_regression": "#077746", "random_forest": "#1ED760"}
TIE_MARGIN = 0.01  # descriptive F1 difference, chosen before examining the results


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fitted_model_source(path):
    """Reassemble and check the exact gzip bytes stored in GitHub-sized parts."""
    path = Path(path)
    if path.suffix != ".json":
        return path
    manifest = json.loads(path.read_text())
    require(manifest.get("format") == "lossless-gzip-parts-v1", "Unknown fitted-model manifest format")
    chunks = []
    for part in manifest["parts"]:
        part_path = (path.parent / part["path"]).resolve()
        require(part_path.is_relative_to(path.parent.resolve()), "Model part leaves its model directory")
        content = part_path.read_bytes()
        require(len(content) == part["bytes"] and hashlib.sha256(content).hexdigest() == part["sha256"],
                f"Model part checksum mismatch: {part['path']}")
        chunks.append(content)
    payload = b"".join(chunks)
    require(len(payload) == manifest["compressed_bytes"] and hashlib.sha256(payload).hexdigest() == manifest["compressed_sha256"],
            "Reassembled fitted-model checksum mismatch")
    return io.BytesIO(payload)


def fitted_model_storage(path):
    path = Path(path)
    if path.suffix == ".json":
        manifest = json.loads(path.read_text())
        return {"storage_sha256": manifest["compressed_sha256"], "bytes": manifest["compressed_bytes"],
                "storage_format": manifest["format"], "manifest_sha256": sha256(path)}
    return {"storage_sha256": sha256(path), "bytes": path.stat().st_size}


def fitted_model_sha256(path):
    """Hash the original model bytes, regardless of lossless gzip storage."""
    path = Path(path)
    source = fitted_model_source(path)
    opener = gzip.open if path.suffix in {".gz", ".json"} else open
    digest = hashlib.sha256()
    with opener(source, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_forest_path():
    """The Git repository stores the exact original model as a gzip file."""
    folder = Path(__file__).resolve().parent / "models"
    manifest = folder / "random_forest.joblib.gz.manifest.json"
    if manifest.is_file():
        return manifest
    compressed = folder / "random_forest.joblib.gz"
    return compressed if compressed.is_file() else folder / "random_forest.joblib"


def validate_predictions(frame, test, classes):
    """Align by unique track ID, check truth/metadata, then name-order probabilities.

    A file with no probabilities is valid for label metrics. A partial or corrupt
    probability matrix fails loudly, instead of producing a misleading ROC-AUC.
    """
    required = {"track_id", "broad_genre", "predicted"}
    require(required <= set(frame), f"Missing columns: {required - set(frame)}")
    require(frame.track_id.notna().all(), "Missing prediction track ID")
    require(frame.track_id.is_unique, "Duplicate prediction track ID")
    require(test.track_id.is_unique, "Duplicate reference test track ID")
    require(set(frame.track_id) == set(test.track_id), "Prediction IDs do not equal the held-out test IDs")
    same_order = frame.track_id.tolist() == test.track_id.tolist()
    aligned = frame.set_index("track_id").loc[test.track_id].reset_index()
    require(aligned.broad_genre.tolist() == test.broad_genre.tolist(), "True labels disagree with clean_songs.csv")
    require(aligned.predicted.isin(classes).all(), "Unknown or missing predicted class")
    for col in ["artists", "track_name"]:
        if col in aligned:
            require(aligned[col].fillna("").tolist() == test[col].fillna("").tolist(), f"{col} metadata mismatch")
    pcols = [f"proba_{g}" for g in classes]
    found = [c for c in frame if c.startswith("proba_")]
    probability = None
    if found:
        require(set(found) == set(pcols), "Probability columns must name every class exactly once")
        probability = aligned[pcols].to_numpy(dtype=float)
        require(np.isfinite(probability).all(), "Non-finite probability")
        require(((probability >= 0) & (probability <= 1)).all(), "Probability outside [0, 1]")
        require(np.allclose(probability.sum(axis=1), 1, atol=1e-8, rtol=0), "Probability rows do not sum to 1")
        # Accept an argmax tie only when the recorded class is one of the maxima.
        chosen = np.array([classes.index(g) for g in aligned.predicted])
        require(np.allclose(probability[np.arange(len(aligned)), chosen], probability.max(axis=1), atol=1e-12, rtol=0),
                "Predicted label disagrees with maximum probability")
    return aligned, probability, {"rows": len(aligned), "original_order_matches": same_order,
                                   "alignment": "track_id", "probability_columns": pcols if found else []}


def artist_tokens(value):
    if pd.isna(value):
        return set()
    return {s.strip() for s in str(value).split(";") if s.strip()}


def input_audit(root, common, songs):
    train = songs[songs.split == "train"].copy()
    test = songs[songs.split == "test"].copy().reset_index(drop=True)
    require(songs.track_id.notna().all() and songs.track_id.is_unique, "Missing/duplicate song IDs")
    require(set(songs.split) == {"train", "test"}, "Unexpected split values")
    require(set(songs.broad_genre) == set(common.GENRES), "Reference classes disagree with common.GENRES")
    require(songs[common.FEATURES].notna().all().all(), "Missing audio feature")
    require(set(train.fold) == set(range(5)) and test.fold.eq(-1).all(), "Unexpected fold assignment")
    groups_train = set(train.artists.fillna("unknown"))
    groups_test = set(test.artists.fillna("unknown"))
    require(not groups_train & groups_test, "Exact artist credit appears in train and test")
    fold_overlap = {}
    for fold in range(5):
        a = set(train.loc[train.fold != fold, "artists"].fillna("unknown"))
        b = set(train.loc[train.fold == fold, "artists"].fillna("unknown"))
        fold_overlap[str(fold)] = len(a & b)
        require(not a & b, f"Exact artist credit crosses training fold {fold}")
    individual_train = set().union(*train.artists.map(artist_tokens))
    individual_test = set().union(*test.artists.map(artist_tokens))
    test["individual_artist_overlap"] = test.artists.map(lambda x: bool(artist_tokens(x) & individual_train))
    files = ["0_foundation.ipynb", "common.py", "dataset.csv", "clean_songs.csv",
             "predictions/logistic_regression.csv", "predictions/random_forest.csv"]
    try:
        top = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.DEVNULL).strip()
        if Path(top).resolve() != root.resolve():
            raise ValueError("The archive is inside an unrelated parent Git repository")
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        commit_source = "checkout Git history"
    except (OSError, subprocess.CalledProcessError, ValueError):
        receipt = Path(__file__).resolve().parent / "DELIVERY.json"
        commit = json.loads(receipt.read_text()).get("upstream_commit", "unavailable") if receipt.exists() else "unavailable"
        commit_source = "archive delivery receipt; current input hashes are recorded separately"
    audit = {"repository_commit": commit, "files_sha256": {f: sha256(root / f) for f in files if (root / f).exists()},
             "commit_provenance": commit_source,
             "total_songs": len(songs), "train_songs": len(train), "test_songs": len(test),
             "classes_probability_order": common.GENRES, "feature_columns": common.FEATURES,
             "exact_artist_credit_overlap": 0, "training_fold_artist_credit_overlap": fold_overlap,
             "individual_artist_token_matching": "semicolon-delimited, whitespace-trimmed, case-sensitive credited names; no identity resolution",
             "shared_individual_artist_tokens": len(individual_train & individual_test),
             "test_tracks_with_individual_artist_overlap": int(test.individual_artist_overlap.sum()),
             "missing_predictors": 0, "duplicate_track_ids": 0,
             "raw_rows": len(pd.read_csv(root / "dataset.csv", usecols=["track_id"])),
             "limitations": [
                 "Groups use full artist credit strings. Individual artists can occur across splits through collaborations.",
                 "Foundation comments describe genre exclusions after weak rock performance. Historical test independence cannot be established from these files.",
                 "The legacy 70.5% notebook uses different data/splitting and is not comparable."]}
    audit["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                            "platform": platform.platform()}
    return train, test, audit


def metric_rows(y, predicted, probability, classes, name):
    p, r, f, support = precision_recall_fscore_support(y, predicted, labels=classes, zero_division=0)
    auc = (roc_auc_score(y, probability, labels=classes, multi_class="ovr", average="macro")
           if probability is not None else np.nan)
    overall = dict(model=name, accuracy=accuracy_score(y, predicted), macro_precision=p.mean(),
                   macro_recall=r.mean(), macro_f1=f.mean(), roc_auc_ovr_macro=auc,
                   correct=int(np.sum(np.asarray(y) == np.asarray(predicted))), test_songs=len(y))
    per = [dict(model=name, genre=g, precision=p[i], recall=r[i], f1=f[i], support=int(support[i]))
           for i, g in enumerate(classes)]
    return overall, per


def bootstrap_difference(test, lr, rf, classes, repetitions=2000):
    """Paired cluster bootstrap, resampling full artist credit groups.

    Both models always receive the same sampled tracks. These intervals describe
    this split and the credit-string grouping. Collaborator overlap remains a
    limitation; the interval is not evidence of new-artist generalization.
    """
    group, unique = pd.factorize(test.artists.fillna("unknown"), sort=True)
    class_number = {g: i for i, g in enumerate(classes)}
    truth = test.broad_genre.map(class_number).to_numpy()
    cubes = []
    for prediction in [lr, rf]:
        prednum = np.array([class_number[x] for x in prediction])
        cube = np.zeros((len(unique), len(classes), len(classes)), dtype=np.int32)
        np.add.at(cube, (group, truth, prednum), 1)
        cubes.append(cube)
    rng = np.random.default_rng(4241)
    values = []
    for _ in range(repetitions):
        sampled = rng.integers(0, len(unique), len(unique))
        metrics = []
        for cube in cubes:
            cm = cube[sampled].sum(axis=0)
            denom = cm.sum(axis=0) + cm.sum(axis=1)
            f1 = np.divide(2 * np.diag(cm), denom, out=np.zeros(len(classes)), where=denom != 0).mean()
            metrics.append((np.trace(cm) / cm.sum(), f1))
        values.append(np.array(metrics[1]) - np.array(metrics[0]))
    arr = np.array(values)
    return [{"metric": name, "low_95": np.quantile(arr[:, i], .025),
             "high_95": np.quantile(arr[:, i], .975), "resamples": repetitions,
             "unit": "full artist credit string", "seed": 4241}
            for i, name in enumerate(["accuracy", "macro_f1"])]


def verify_model_outputs(model, X, y, expected_labels, expected_proba, classes):
    """Gate interpretation on feature/class order and the complete saved output."""
    require(list(model.classes_) == list(classes), "Model class order differs from the saved probability order")
    if hasattr(model, "feature_names_in_"):
        require(list(model.feature_names_in_) == list(X.columns), "Model feature order differs from the shared predictors")
    predicted = model.predict(X)
    actual_proba = model.predict_proba(X)
    agreement = float(np.mean(predicted == expected_labels))
    max_difference = (float(np.max(np.abs(actual_proba - expected_proba))) if expected_proba is not None else None)
    return predicted, dict(label_agreement=agreement, probability_max_abs_difference=max_difference,
                           model_test_accuracy=float(np.mean(predicted == y)),
                           differing_labels=int(np.sum(predicted != expected_labels)),
                           matches_saved_outputs=agreement == 1 and (max_difference is None or max_difference < 1e-8))


def refit_and_explain(common, train, test, predictions, probabilities, out, repeats, rf_model=None):
    """Verify a fixed LR refit and the supplied fitted RF before interpretation."""
    complexity, importance, raw_importance, reproduction = [], [], [], []
    for name in ["logistic_regression", "random_forest"]:
        load_seconds = np.nan
        fit_seconds = np.nan
        source = "fixed_recipe_refit_on_train"
        if name == "random_forest" and rf_model is not None:
            source = "confirmed_original_fitted_model"
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", InconsistentVersionWarning)
                model = joblib.load(fitted_model_source(rf_model))
            versions = sorted({getattr(w.message, "original_sklearn_version", "unknown")
                               for w in caught if isinstance(w.message, InconsistentVersionWarning)})
            require(not versions, f"Fitted model requires scikit-learn {versions}; local version is {sklearn.__version__}. Use the pinned environment.")
            load_seconds = time.perf_counter() - start
            sample_count = getattr(model, "_n_samples", None)
            require(sample_count is None or sample_count == len(train), "Fitted forest training sample count differs from the shared training split")
        else:
            model = common.MODELS[name]()
        # A single worker makes the local batch-prediction measurements comparable.
        if name == "random_forest":
            model.set_params(n_jobs=1)
        with threadpool_limits(limits=1):
            if source == "fixed_recipe_refit_on_train":
                start = time.perf_counter()
                model.fit(train[common.FEATURES], train.broad_genre)
                fit_seconds = time.perf_counter() - start
            predicted, check = verify_model_outputs(model, test[common.FEATURES], test.broad_genre,
                                                     predictions[name], probabilities[name], common.GENRES)
            reproduction.append(dict(model=name, source=source, **check))
            if not check["matches_saved_outputs"]:
                mismatch = test.loc[predicted != predictions[name], ["track_id", "artists", "track_name", "broad_genre"]].copy()
                mismatch["saved_prediction"] = predictions[name][predicted != predictions[name]]
                mismatch["refit_prediction"] = predicted[predicted != predictions[name]]
                mismatch.to_csv(out / f"tables/refit_mismatches_{name}.csv", index=False)
                print(f"WARNING: {name} differs from saved outputs; skipping its feature interpretation.", flush=True)
                continue
            (out / f"tables/refit_mismatches_{name}.csv").unlink(missing_ok=True)
            # Warm up, then report median batch-prediction time for the same 9,009 tracks.
            model.predict(test[common.FEATURES])
            elapsed = []
            for _ in range(5):
                start = time.perf_counter()
                model.predict(test[common.FEATURES])
                elapsed.append(time.perf_counter() - start)
            row = dict(model=name, source=source, fit_seconds=fit_seconds, load_seconds=load_seconds,
                       prediction_batch_seconds_median=np.median(elapsed),
                       prediction_repeats=5, workers=1, serialized_bytes=len(pickle.dumps(model, protocol=5)))
            if name == "random_forest":
                row.update(trees=len(model.estimators_), tree_nodes=sum(x.tree_.node_count for x in model.estimators_),
                           median_depth=float(np.median([x.tree_.max_depth for x in model.estimators_])),
                           max_depth=max(x.tree_.max_depth for x in model.estimators_))
                pd.DataFrame({"feature": common.FEATURES, "gini_share": model.feature_importances_}).sort_values(
                    "gini_share", ascending=False).to_csv(out / "tables/verified_rf_gini_importance.csv", index=False)
            else:
                lr = model.named_steps["logisticregression"]
                row.update(coefficients=int(lr.coef_.size), intercepts=int(lr.intercept_.size))
            complexity.append(row)
            print(f"{name}: verified saved labels/probabilities ({source}); computing {repeats} permutation repeats.", flush=True)
            if repeats:
                imp = permutation_importance(model, test[common.FEATURES], test.broad_genre,
                                             scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"},
                                             n_repeats=repeats, random_state=4241, n_jobs=1)
                for score, result in imp.items():
                    for j, feature in enumerate(common.FEATURES):
                        importance.append(dict(model=name, metric=score, feature=feature,
                                               mean_score_drop=result.importances_mean[j],
                                               sd_score_drop=result.importances_std[j], repeats=repeats))
                        raw_importance.extend(dict(model=name, metric=score, feature=feature, repeat=k,
                                                   score_drop=float(v)) for k, v in enumerate(result.importances[j]))
    pd.DataFrame(reproduction).to_csv(out / "tables/model_verification.csv", index=False)
    pd.DataFrame(complexity).to_csv(out / "tables/complexity.csv", index=False)
    if importance:
        pd.DataFrame(importance).to_csv(out / "tables/permutation_importance.csv", index=False)
        pd.DataFrame(raw_importance).to_csv(out / "tables/permutation_repeats.csv", index=False)
    return complexity, importance, reproduction


def charts(out, overall, per_genre, probabilities, y, classes, importance):
    """Publication/export charts are data plots, with CSV sources beside them."""
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Avenir Next", "DejaVu Sans"],
                         "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "#F4F4F0", "axes.facecolor": "#F4F4F0"})
    def save(fig, name):
        fig.savefig(out / f"charts/{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(out / f"charts/{name}.svg", bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.1))
    positions = np.arange(4)
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    for j, row in enumerate(overall):
        v = [row[m] for m in metrics]
        bars = ax.bar(positions + (j - 1) * .24, v, .22, color=COLORS[row["model"]], label=NAMES[row["model"]])
        ax.bar_label(bars, labels=[f"{x:.3f}" for x in v], padding=3, fontsize=10)
    ax.set(xticks=positions, xticklabels=["Accuracy", "Macro precision", "Macro recall", "Macro F1"], ylim=(0, .78),
           title="The frozen Random Forest improves every overall metric")
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=10)
    save(fig, "overall_metrics")
    p = pd.DataFrame(per_genre)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, metric in zip(axes, ["precision", "recall", "f1"]):
        for j, name in enumerate(["logistic_regression", "random_forest"]):
            v = p[p.model == name].set_index("genre").loc[ORDER, metric]
            ax.barh(np.arange(5) + (j - .5) * .32, v, .30, color=COLORS[name], label=NAMES[name])
        ax.set(yticks=range(5), yticklabels=[DISPLAY[g] for g in ORDER], xlim=(0, 1), title=metric.capitalize())
        ax.grid(axis="x", alpha=.15)
    axes[0].invert_yaxis()
    axes[-1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Per-genre comparison on the same test tracks", fontweight="bold")
    fig.tight_layout()
    save(fig, "per_genre_metrics")
    fig, ax = plt.subplots(figsize=(10, 5.3))
    for j, name in enumerate(["logistic_regression", "random_forest"]):
        v = p[p.model == name].set_index("genre").loc[ORDER, "f1"]
        b = ax.barh(np.arange(5) + (j - .5) * .33, v, .30, color=COLORS[name], label=NAMES[name])
        ax.bar_label(b, labels=[f"{x:.3f}" for x in v], padding=4, fontsize=10)
    ax.set(yticks=range(5), yticklabels=[DISPLAY[g] for g in ORDER], xlim=(0, .95), xlabel="F1 score", title="Random Forest gains F1 in every genre")
    ax.invert_yaxis(); ax.legend(frameon=False, loc="lower right")
    save(fig, "genre_f1")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, name in zip(axes, ["logistic_regression", "random_forest"]):
        proba = probabilities[name]
        if proba is None:
            continue
        for g in ORDER:
            i = classes.index(g)
            fpr, tpr, _ = roc_curve(np.asarray(y) == g, proba[:, i])
            ax.plot(fpr, tpr, label=DISPLAY[g], linewidth=1.6)
        ax.plot([0, 1], [0, 1], "--", color="#A9ABA5", linewidth=1)
        ax.set(title=NAMES[name], xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
    axes[-1].legend(frameon=False, fontsize=9)
    fig.tight_layout(); save(fig, "roc_ovr")
    if importance:
        z = pd.DataFrame(importance); z = z[z.metric == "macro_f1"]
        features = z.groupby("feature").mean_score_drop.max().sort_values().tail(10).index
        fig, ax = plt.subplots(figsize=(10, 6))
        available = [name for name in ["logistic_regression", "random_forest"] if name in set(z.model)]
        for j, name in enumerate(available):
            v = z[z.model == name].set_index("feature").reindex(features)
            ax.barh(np.arange(len(features)) + (j - .5) * .33, v.mean_score_drop, .30,
                    xerr=v.sd_score_drop, label=NAMES[name], color=COLORS[name], capsize=2)
        ax.set(yticks=range(len(features)), yticklabels=features, xlabel="Drop in test macro F1 after shuffling one feature",
               title=("Common feature comparison: permutation importance" if len(available) == 2
                      else "Verified Logistic Regression permutation importance"))
        ax.axvline(0, color="#A9ABA5", linewidth=1); ax.legend(frameon=False)
        fig.tight_layout(); save(fig, "permutation_importance")


def run(root=None, output=None, refit=True, repeats=5, bootstrap=2000, rf_model=None):
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    out = Path(output or Path(__file__).resolve().parent / "results").resolve()
    for folder in [out, out / "tables", out / "charts"]:
        folder.mkdir(parents=True, exist_ok=True)
    # Remove only this program's optional generated artifacts, so a metrics-only
    # refresh cannot silently display interpretation left over from an older run.
    if not refit:
        for filename in ["model_verification.csv", "complexity.csv", "permutation_importance.csv", "permutation_repeats.csv", "verified_rf_gini_importance.csv"]:
            (out / "tables" / filename).unlink(missing_ok=True)
        for extension in ["png", "svg"]:
            (out / "charts" / f"permutation_importance.{extension}").unlink(missing_ok=True)
    spec = importlib.util.spec_from_file_location("project_common", root / "common.py")
    common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
    songs = pd.read_csv(root / "clean_songs.csv")
    train, test, audit = input_audit(root, common, songs)
    classes = list(common.GENRES)
    require(classes == sorted(classes), "sklearn multiclass ROC-AUC requires lexical class order")
    predictions, probabilities = {}, {}
    missing = [str(root / f"predictions/{n}.csv") for n in ["logistic_regression", "random_forest"]
               if not (root / f"predictions/{n}.csv").exists()]
    if missing:
        audit["missing_inputs"] = missing
        (out / "audit.json").write_text(json.dumps(audit, indent=2))
        raise FileNotFoundError("Final comparison awaits: " + ", ".join(missing))
    audit["prediction_validation"] = {}
    for name in ["logistic_regression", "random_forest"]:
        frame, proba, status = validate_predictions(pd.read_csv(root / f"predictions/{name}.csv"), test, classes)
        predictions[name] = frame.predicted.to_numpy(); probabilities[name] = proba
        audit["prediction_validation"][name] = status
    majority = train.broad_genre.value_counts().idxmax()
    predictions = {"baseline": np.repeat(majority, len(test)), **predictions}
    base_probability = np.zeros((len(test), len(classes)))
    base_probability[:, classes.index(majority)] = 1
    probabilities = {"baseline": base_probability, **probabilities}
    audit["baseline_class_chosen_from_train"] = majority
    y = test.broad_genre.to_numpy()
    overall, per_genre = [], []
    for name, pred in predictions.items():
        row, per = metric_rows(y, pred, probabilities[name], classes, name)
        overall.append(row); per_genre.extend(per)
        cm = confusion_matrix(y, pred, labels=classes)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / f"tables/confusion_{name}.csv", index_label="actual")
    pd.DataFrame(overall).to_csv(out / "tables/overall_metrics.csv", index=False)
    pd.DataFrame(per_genre).to_csv(out / "tables/per_genre_metrics.csv", index=False)
    counts = pd.crosstab(songs.broad_genre, songs.split).reindex(classes)
    counts.to_csv(out / "tables/class_counts.csv", index_label="genre")
    lr = predictions["logistic_regression"]; rf = predictions["random_forest"]
    correct_lr, correct_rf = lr == y, rf == y
    paired = test.copy()
    paired["lr_predicted"] = lr; paired["rf_predicted"] = rf
    paired["lr_correct"] = correct_lr; paired["rf_correct"] = correct_rf
    paired["outcome"] = np.select([correct_lr & correct_rf, ~correct_lr & correct_rf, correct_lr & ~correct_rf],
                                   ["both_correct", "rf_only_correct", "lr_only_correct"], default="both_wrong")
    paired["models_disagree"] = lr != rf
    for name, proba in probabilities.items():
        if name == "baseline" or proba is None: continue
        prefix = "lr" if name == "logistic_regression" else "rf"
        paired[f"{prefix}_confidence"] = proba.max(axis=1)
        paired[f"{prefix}_true_probability"] = proba[np.arange(len(test)), [classes.index(g) for g in y]]
    paired.to_csv(out / "tables/paired_tracks.csv", index=False)
    disagreements = paired[paired.models_disagree]
    disagreements.to_csv(out / "tables/disagreements.csv", index=False)
    paired.groupby(["broad_genre", "lr_predicted", "rf_predicted", "outcome"]).size().reset_index(name="songs").sort_values("songs", ascending=False).to_csv(
        out / "tables/prediction_patterns.csv", index=False)
    paired.outcome.value_counts().rename_axis("outcome").reset_index(name="songs").to_csv(out / "tables/paired_outcomes.csv", index=False)
    per_pair = pd.crosstab(paired.broad_genre, paired.outcome).reindex(classes).fillna(0).astype(int)
    per_pair["net_extra_correct_rf"] = per_pair.get("rf_only_correct", 0) - per_pair.get("lr_only_correct", 0)
    per_pair.to_csv(out / "tables/paired_outcomes_by_genre.csv", index_label="genre")
    p = pd.DataFrame(per_genre)
    a = p[p.model == "logistic_regression"].set_index("genre")
    b = p[p.model == "random_forest"].set_index("genre")
    delta = pd.DataFrame({f"delta_{m}": b[m] - a[m] for m in ["precision", "recall", "f1"]})
    delta["f1_comparison"] = np.select([delta.delta_f1 > TIE_MARGIN, delta.delta_f1 < -TIE_MARGIN],
                                       ["Random Forest wins", "Logistic Regression wins"], default="Approximately tied")
    delta["descriptive_tie_margin_f1"] = TIE_MARGIN
    delta.to_csv(out / "tables/genre_deltas.csv", index_label="genre")
    # Representative cases: median true-class probability advantage in each genre/outcome,
    # rather than cherry-picking the most extreme success.
    examples = []
    if "lr_true_probability" in paired and "rf_true_probability" in paired:
        paired["rf_minus_lr_true_probability"] = paired.rf_true_probability - paired.lr_true_probability
        for genre in ORDER:
            for outcome in ["rf_only_correct", "lr_only_correct"]:
                z = paired[(paired.broad_genre == genre) & (paired.outcome == outcome)].sort_values(["rf_minus_lr_true_probability", "track_id"])
                if len(z): examples.append(z.iloc[len(z) // 2])
    if examples:
        pd.DataFrame(examples).to_csv(out / "tables/representative_tracks.csv", index=False)
    subset_rows = []
    for overlap in [False, True]:
        mask = test.individual_artist_overlap.to_numpy() == overlap
        if mask.any():
            for name in ["logistic_regression", "random_forest"]:
                row, _ = metric_rows(y[mask], predictions[name][mask], None, classes, name)
                row["individual_artist_overlap"] = overlap; subset_rows.append(row)
    pd.DataFrame(subset_rows).to_csv(out / "tables/artist_overlap_subgroups.csv", index=False)
    if bootstrap:
        intervals = pd.DataFrame(bootstrap_difference(test, lr, rf, classes, bootstrap))
        om = pd.DataFrame(overall).set_index("model")
        intervals["observed_difference_rf_minus_lr"] = intervals.metric.map(
            lambda m: om.loc["random_forest", m] - om.loc["logistic_regression", m])
        intervals.to_csv(out / "tables/paired_bootstrap.csv", index=False)
    complexity, importance, reproduction = [], [], []
    model_path = Path(rf_model).resolve() if rf_model else default_forest_path()
    if rf_model:
        require(model_path.is_file(), f"Fitted forest does not exist: {model_path}")
    if not model_path.is_file():
        model_path = None
    if refit:
        complexity, importance, reproduction = refit_and_explain(common, train, test, predictions, probabilities, out, repeats, model_path)
    audit.update(model_evaluation_requested=refit, model_verification=reproduction, permutation_repeats=repeats if refit else 0,
                 permutation_scope="held-out test, frozen verified models, descriptive interpretation only",
                 bootstrap_repetitions=bootstrap, tie_margin_f1=TIE_MARGIN,
                 source_file_policy="shared data, model recipes, and saved predictions were not modified",
                 fitted_rf_model=({"path": str(Path("models") / model_path.name) if not rf_model else str(model_path),
                                   "sha256": fitted_model_sha256(model_path),
                                   "hash_scope": "original model bytes after optional gzip decompression",
                                   **fitted_model_storage(model_path)} if model_path else None))
    audit["unresolved_dependencies"] = (["The original fitted Random Forest and its training environment, or a refit that reproduces the saved labels and probabilities, are needed for an official forest permutation comparison and measured complexity."]
                                        if refit and any(not r["matches_saved_outputs"] for r in reproduction) else [])
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    charts(out, overall, per_genre, probabilities, y, classes, importance)
    print(pd.DataFrame(overall).to_string(index=False), flush=True)
    print("\nPaired outcomes:", paired.outcome.value_counts().to_dict(), flush=True)
    print(f"\nSaved verified outputs to {out}", flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-refit", action="store_true")
    parser.add_argument("--rf-model", type=Path, help="Confirmed fitted forest; defaults to the checked gzip-parts manifest, with .gz and .joblib fallback")
    parser.add_argument("--permutations", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    require(args.permutations >= 0 and args.bootstrap >= 0, "Repeat counts must be nonnegative")
    run(args.root, args.output, not args.skip_refit, args.permutations, args.bootstrap, args.rf_model)

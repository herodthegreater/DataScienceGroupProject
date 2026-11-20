"""Audit the newly supplied CSVs and extract recorded RF notebook evidence.

The teammate notebook is read as data, not executed. Its CV/feature outputs are
reported as teammate-provided evidence, separate from our verified test metrics.
"""
from pathlib import Path
import hashlib
import json
import re
import pandas as pd
import numpy as np

from evaluate import validate_predictions, default_forest_path, fitted_model_sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT = HERE / "teammate_inputs"
RESULTS = HERE / "results"
CLASSES = ["electronic", "hiphop/rnb", "latin", "pop", "rock"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_outputs(cell):
    return "\n".join("".join(o.get("text") or o.get("data", {}).get("text/plain", []))
                     for o in cell.get("outputs", []))


def audit_inputs():
    songs = pd.read_csv(ROOT / "clean_songs.csv")
    test = songs[songs.split == "test"].reset_index(drop=True)
    receipt = {"status": "verified", "shared_test_songs": len(test), "files": {},
               "excluded_earlier_files": ["Downloads/requirements.txt", "Downloads/dataset.csv"],
               "earlier_files_policy": "The user subsequently confirmed the fitted model. The other earlier downloads remain unnecessary; the shared repository data and independently pinned evaluation environment are used."}
    for model in ["logistic_regression", "random_forest"]:
        path = INPUT / f"{model}.csv"
        frame = pd.read_csv(path)
        _, _, validation = validate_predictions(frame, test, CLASSES)
        old = ROOT / f"predictions/{model}.csv"
        reference = pd.read_csv(old)
        same = frame.equals(reference)
        if not same:
            raise ValueError("New predictions changed. Recalculate the comparison before updating the report.")
        receipt["files"][path.name] = {
            "sha256": digest(path), "original_repository_sha256": digest(old),
            "parsed_cells_equal_repository": same,
            "only_line_endings_differ": path.read_bytes().replace(b"\r\n", b"\n") == old.read_bytes(),
            "max_probability_difference": float(np.max(np.abs(frame.filter(like="proba_").to_numpy() - reference.filter(like="proba_").to_numpy()))),
            "test_accuracy": float(frame.predicted.eq(frame.broad_genre).mean()),
            "validation": validation,
        }
    path = INPUT / "4_random_forest.ipynb"
    notebook = json.loads(path.read_text())
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    features_cell = next(c for c in code_cells if "rf_features.feature_importances_" in "".join(c["source"]))
    rows = re.findall(r"^([a-z_]+)\s+([0-9]+\.[0-9]+)\s*$", text_outputs(features_cell), re.M)
    feature_table = pd.DataFrame(rows, columns=["feature", "gini_share"])
    feature_table.gini_share = feature_table.gini_share.astype(float)
    feature_table["rank"] = range(1, len(feature_table) + 1)
    feature_table["method"] = "training-fitted RF Gini impurity reduction"
    feature_table["provenance"] = "teammate notebook recorded output, 6 decimal places; not rerun"
    if len(feature_table) != 14 or abs(feature_table.gini_share.sum() - 1) > 1e-5:
        raise ValueError("RF importance output is incomplete or inconsistent")
    feature_table.to_csv(RESULTS / "tables/teammate_rf_gini_importance.csv", index=False)

    depth_cell = next(c for c in code_cells if "Depth {depth}:" in "".join(c["source"]))
    depths = re.findall(r"Depth (5|10|15|20|None): Train=([0-9.]+), CV=([0-9.]+)", text_outputs(depth_cell))
    depth_table = pd.DataFrame(depths, columns=["max_depth", "mean_fold_train_accuracy", "mean_fold_validation_accuracy"])
    for col in depth_table.columns[1:]:
        depth_table[col] = depth_table[col].astype(float)
    depth_table["provenance"] = "teammate notebook recorded output, 3 decimal places; not rerun"
    depth_table.to_csv(RESULTS / "tables/teammate_rf_depth_cv.csv", index=False)
    weight_cell = next(c for c in code_cells if "Mean validation macro F1:" in "".join(c["source"]))
    weights = re.findall(r"Class weight: (None|balanced|balanced_subsample)\s+Mean validation accuracy: ([0-9.]+)\s+Mean validation macro F1: ([0-9.]+)", text_outputs(weight_cell))
    weight_table = pd.DataFrame(weights, columns=["class_weight", "mean_validation_accuracy", "mean_validation_macro_f1"])
    for col in weight_table.columns[1:]:
        weight_table[col] = weight_table[col].astype(float)
    weight_table["provenance"] = "teammate notebook recorded output, 3 decimal places; not rerun"
    weight_table.to_csv(RESULTS / "tables/teammate_rf_weight_cv.csv", index=False)
    if len(depth_table) != 5 or len(weight_table) != 3:
        raise ValueError("CV outputs could not be extracted completely")

    lr = pd.read_csv(RESULTS / "tables/permutation_importance.csv")
    lr = lr[(lr.model == "logistic_regression") & (lr.metric == "macro_f1")].sort_values("mean_score_drop", ascending=False).reset_index(drop=True)
    comparison = pd.DataFrame({
        "rank": range(1, 15),
        "lr_feature": lr.feature,
        "lr_test_macro_f1_drop": lr.mean_score_drop,
        "rf_feature": feature_table.feature,
        "rf_training_gini_share": feature_table.gini_share,
    })
    comparison["interpretation"] = "Separate within-model rankings. Different definitions and scales; magnitudes are not comparable."
    comparison.to_csv(RESULTS / "tables/feature_rankings_different_measures.csv", index=False)
    receipt["files"][path.name] = {
        "sha256": digest(path), "cells": len(notebook["cells"]), "code_cells": len(code_cells),
        "notebook_python_metadata": notebook.get("metadata", {}).get("language_info", {}).get("version"),
        "contains_fitted_model": False,
        "contains_training_package_versions": False,
        "analysis_scope": "fixed-recipe training CV, class-weighting OOF counts, Gini importance, saved test evaluation",
        "code_inspected": True, "experiments_rerun": False,
    }
    receipt["resolved"] = ["Prediction outputs reconfirmed", "RF depth/weighting code supplied", "RF native Gini ranking supplied"]
    model_path = default_forest_path()
    verification_path = RESULTS / "tables/model_verification.csv"
    model_verified = False
    if model_path.exists() and verification_path.exists():
        verified = pd.read_csv(verification_path).set_index("model")
        model_verified = bool(verified.loc["random_forest", "matches_saved_outputs"])
        receipt["files"]["models/" + model_path.name] = {"sha256": fitted_model_sha256(model_path), "user_confirmed": True, "matches_saved_outputs": model_verified}
    receipt["remaining"] = [] if model_verified else ["Verified fitted RF model for common test permutation importance and measured complexity"]
    if model_verified:
        receipt["resolved"].append("Confirmed original fitted forest passes complete label/probability check")
        rf = pd.read_csv(RESULTS / "tables/verified_rf_gini_importance.csv").set_index("feature")
        native = feature_table.set_index("feature")
        receipt["rf_gini_matches_notebook_to_six_decimals"] = bool(np.allclose(
            rf.loc[native.index, "gini_share"], native.gini_share, atol=0.0000005, rtol=0))
        common = pd.read_csv(RESULTS / "tables/permutation_importance.csv")
        common = common[common.metric == "macro_f1"].pivot(index="feature", columns="model", values="mean_score_drop")
        common.columns = ["lr_macro_f1_drop" if c == "logistic_regression" else "rf_macro_f1_drop" for c in common.columns]
        common["rf_minus_lr_drop"] = common.rf_macro_f1_drop - common.lr_macro_f1_drop
        common.sort_values("rf_macro_f1_drop", ascending=False).to_csv(RESULTS / "tables/common_permutation_comparison.csv", index_label="feature")
    (RESULTS / "teammate_input_audit.json").write_text(json.dumps(receipt, indent=2))
    print("New CSVs verified. RF notebook evidence extracted without execution.")
    print(feature_table.head(5).to_string(index=False))
    return receipt


if __name__ == "__main__":
    audit_inputs()

"""Small safeguards for the comparison's most consequential failure modes."""
import unittest
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from evaluate import validate_predictions, verify_model_outputs, artist_tokens, fitted_model_sha256


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.classes = ["a", "b"]
        self.test = pd.DataFrame({"track_id": ["song1", "song2"], "broad_genre": ["a", "b"],
                                  "artists": ["Artist 1", "Artist 2"], "track_name": ["One", "Two"]})
        self.pred = self.test.assign(predicted=["a", "b"], proba_a=[.8, .2], proba_b=[.2, .8])

    def test_reorders_by_id_and_probability_column_name(self):
        source = self.pred.iloc[::-1][["track_id", "broad_genre", "predicted", "proba_b", "proba_a"]]
        aligned, probability, status = validate_predictions(source, self.test, self.classes)
        self.assertEqual(aligned.track_id.tolist(), ["song1", "song2"])
        np.testing.assert_allclose(probability, [[.8, .2], [.2, .8]])
        self.assertFalse(status["original_order_matches"])

    def test_rejects_duplicates(self):
        source = self.pred.copy(); source.loc[1, "track_id"] = "song1"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_predictions(source, self.test, self.classes)

    def test_rejects_wrong_test_song(self):
        source = self.pred.copy(); source.loc[1, "track_id"] = "other"
        with self.assertRaisesRegex(ValueError, "held-out"):
            validate_predictions(source, self.test, self.classes)

    def test_rejects_wrong_truth(self):
        source = self.pred.copy(); source.loc[1, "broad_genre"] = "a"
        with self.assertRaisesRegex(ValueError, "True labels"):
            validate_predictions(source, self.test, self.classes)

    def test_rejects_partial_probability_matrix(self):
        with self.assertRaisesRegex(ValueError, "every class"):
            validate_predictions(self.pred.drop(columns="proba_b"), self.test, self.classes)

    def test_rejects_corrupt_probabilities(self):
        for value in [np.nan, -0.2, 1.2, .5]:
            source = self.pred.copy(); source.loc[0, "proba_a"] = value
            with self.assertRaises(ValueError):
                validate_predictions(source, self.test, self.classes)

    def test_rejects_label_probability_disagreement(self):
        source = self.pred.copy(); source.loc[0, "predicted"] = "b"
        with self.assertRaisesRegex(ValueError, "maximum probability"):
            validate_predictions(source, self.test, self.classes)

    def test_label_only_file_can_still_be_compared(self):
        _, probability, _ = validate_predictions(self.pred.drop(columns=["proba_a", "proba_b"]), self.test, self.classes)
        self.assertIsNone(probability)


class ArtistCreditTests(unittest.TestCase):
    def test_preserves_credited_case_while_trimming_collaborators(self):
        self.assertEqual(artist_tokens(" LiSA ; Artist 2 ; "), {"LiSA", "Artist 2"})
        self.assertFalse(artist_tokens("LiSA") & artist_tokens("LISA"))


class ModelStorageTests(unittest.TestCase):
    def make_manifest(self, folder):
        original = b"exact original model bytes"
        compressed = gzip.compress(original, mtime=0)
        part = folder / "part01"
        part.write_bytes(compressed)
        manifest = {"format": "lossless-gzip-parts-v1", "compressed_bytes": len(compressed),
                    "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
                    "parts": [{"path": part.name, "bytes": len(compressed),
                               "sha256": hashlib.sha256(compressed).hexdigest()}]}
        path = folder / "model.json"
        path.write_text(json.dumps(manifest))
        return path, part, original

    def test_split_storage_preserves_original_model_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _, original = self.make_manifest(Path(folder))
            self.assertEqual(fitted_model_sha256(path), hashlib.sha256(original).hexdigest())

    def test_corrupt_model_part_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, part, _ = self.make_manifest(Path(folder))
            part.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                fitted_model_sha256(path)


class FittedModelGuardTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({"first": [0., 1.], "second": [1., 0.]})
        self.classes = ["a", "b"]
        self.y = np.array(["a", "b"])
        self.proba = np.array([[.8, .2], [.1, .9]])
        class SavedModel:
            classes_ = np.array(["a", "b"])
            feature_names_in_ = np.array(["first", "second"])
            def predict(self, X): return np.array(["a", "b"])
            def predict_proba(self, X): return np.array([[.8, .2], [.1, .9]])
        self.model = SavedModel()

    def test_exact_output_passes(self):
        _, check = verify_model_outputs(self.model, self.X, self.y, self.y, self.proba, self.classes)
        self.assertTrue(check["matches_saved_outputs"])

    def test_same_labels_different_probabilities_fail(self):
        expected = self.proba.copy(); expected[0] = [.7, .3]
        _, check = verify_model_outputs(self.model, self.X, self.y, self.y, expected, self.classes)
        self.assertFalse(check["matches_saved_outputs"])
        self.assertEqual(check["label_agreement"], 1.)

    def test_wrong_class_order_fails(self):
        self.model.classes_ = np.array(["b", "a"])
        with self.assertRaisesRegex(ValueError, "class order"):
            verify_model_outputs(self.model, self.X, self.y, self.y, self.proba, self.classes)

    def test_wrong_feature_order_fails(self):
        with self.assertRaisesRegex(ValueError, "feature order"):
            verify_model_outputs(self.model, self.X[["second", "first"]], self.y, self.y, self.proba, self.classes)


if __name__ == "__main__":
    unittest.main()

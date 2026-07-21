import json
import tempfile
import unittest
from pathlib import Path

from temporal_gnn.compare import FAIRNESS_KEYS, compare_results


def make_result(variant, average_precision, roc_auc):
    config = {key: 1 for key in FAIRNESS_KEYS}
    config.update({"dataset": "Wikipedia", "model_variant": variant})
    overall = {"average_precision": average_precision, "roc_auc": roc_auc}
    return {
        "config": config,
        "trainable_parameters": 100 if variant == "baseline" else 120,
        "test": {
            "overall": overall,
            "chronological_bins": [],
            "inactivity_gap_bins": [],
        },
    }


class ComparisonTests(unittest.TestCase):
    def test_reports_enhanced_minus_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "baseline.json"
            enhanced_path = root / "enhanced.json"
            baseline_path.write_text(
                json.dumps(make_result("baseline", 0.70, 0.75))
            )
            enhanced_path.write_text(
                json.dumps(make_result("enhanced", 0.74, 0.77))
            )
            result = compare_results(baseline_path, enhanced_path, root / "out")
            self.assertAlmostEqual(
                result["summary"]["delta"]["average_precision"], 0.04
            )
            self.assertTrue((root / "out" / "comparison.md").exists())

    def test_rejects_uncontrolled_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = make_result("baseline", 0.70, 0.75)
            enhanced = make_result("enhanced", 0.74, 0.77)
            enhanced["config"]["seed"] = 99
            baseline_path = root / "baseline.json"
            enhanced_path = root / "enhanced.json"
            baseline_path.write_text(json.dumps(baseline))
            enhanced_path.write_text(json.dumps(enhanced))
            with self.assertRaisesRegex(ValueError, "seed"):
                compare_results(baseline_path, enhanced_path, root / "out")


if __name__ == "__main__":
    unittest.main()

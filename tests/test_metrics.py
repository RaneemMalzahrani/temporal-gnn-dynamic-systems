import unittest

from temporal_gnn.metrics import inactivity_gap_performance


class TimeAwareMetricTests(unittest.TestCase):
    def test_inactivity_bins_are_sorted_by_gap(self):
        rows = inactivity_gap_performance(
            positive_scores=[0.9, 0.8, 0.7, 0.6],
            negative_scores=[0.1, 0.2, 0.3, 0.4],
            inactivity_gaps=[40, 10, 30, 20],
            n_bins=2,
        )
        self.assertEqual(rows[0]["min_gap"], 10.0)
        self.assertEqual(rows[0]["max_gap"], 20.0)
        self.assertEqual(rows[1]["min_gap"], 30.0)
        self.assertEqual(rows[1]["max_gap"], 40.0)

    def test_rejects_misaligned_gap_values(self):
        with self.assertRaisesRegex(ValueError, "align"):
            inactivity_gap_performance([0.9], [0.1], [], n_bins=1)


if __name__ == "__main__":
    unittest.main()

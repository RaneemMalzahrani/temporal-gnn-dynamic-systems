import unittest

from temporal_gnn.validation import (
    assert_non_decreasing,
    assert_temporal_boundaries,
    chronological_bins,
)


class TemporalValidationTests(unittest.TestCase):
    def test_accepts_strict_chronological_splits(self):
        assert_temporal_boundaries([1, 2, 3], [4, 5], [6, 7])

    def test_accepts_equal_boundary_timestamps(self):
        assert_temporal_boundaries([1, 2], [2, 3], [3, 4])

    def test_rejects_shuffled_events(self):
        with self.assertRaisesRegex(ValueError, "not sorted"):
            assert_non_decreasing([1, 3, 2], "train")

    def test_rejects_future_leakage(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_temporal_boundaries([1, 5], [4, 6], [7, 8])

    def test_chronological_bins_cover_all_events(self):
        self.assertEqual(chronological_bins(8, 4), [0, 0, 1, 1, 2, 2, 3, 3])


if __name__ == "__main__":
    unittest.main()

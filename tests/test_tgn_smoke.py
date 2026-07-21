import math
import unittest

import torch
from torch_geometric.data import TemporalData

from temporal_gnn.config import ExperimentConfig
from temporal_gnn.experiment import TGNExperiment


class TGNIntegrationTests(unittest.TestCase):
    def test_small_temporal_stream_trains_and_evaluates(self):
        src = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        dst = torch.tensor([3, 4, 5, 4, 5, 3, 5, 3, 4, 3, 5, 4])
        timestamps = torch.arange(1, 13)
        messages = torch.arange(48, dtype=torch.float).reshape(12, 4) / 48
        data = TemporalData(src=src, dst=dst, t=timestamps, msg=messages)

        for variant in ("baseline", "enhanced"):
            with self.subTest(variant=variant):
                config = ExperimentConfig(
                    batch_size=4,
                    neighbor_size=3,
                    memory_dim=8,
                    time_dim=8,
                    embedding_dim=8,
                    model_variant=variant,
                    epochs=1,
                    patience=1,
                )
                experiment = TGNExperiment(config, data, torch.device("cpu"))
                loss = experiment.train_epoch(data[:8])
                validation = experiment.evaluate(data[8:10], seed=42)

                self.assertTrue(math.isfinite(loss))
                self.assertEqual(
                    validation["chronological_bins"][0]["events"],
                    1,
                )
                self.assertIn("average_precision", validation["overall"])
                self.assertIn("inactivity_gap_bins", validation)


if __name__ == "__main__":
    unittest.main()

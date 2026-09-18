import unittest
import numpy as np

from emg_v4_core import (CalibrationProfile, GestureDecisionEngine, WirelessStats,
                         assess_signal_quality, compute_calibration, expected_feature_count,
                         validate_model_artifact)
from realtime_pipeline import SampleRingBuffer
from rf_features import extract_window_features_legacy, extract_window_features_optimized


class FakeModel:
    classes_ = np.array([0, 1])
    n_features_in_ = 70
    def predict(self, x): return np.zeros(len(x))


class V4CoreTests(unittest.TestCase):
    def test_packet_gap_and_wrap(self):
        stats = WirelessStats()
        self.assertEqual(stats.observe(100, 1.0), 0)
        self.assertEqual(stats.observe(101, 1.01), 0)
        self.assertEqual(stats.observe(103, 1.02), 1)
        self.assertEqual(stats.packets_missing, 1)

    def test_calibration_and_bad_channel(self):
        rng = np.random.default_rng(4)
        rest = rng.normal(2000, 3, (300, 2)); flex = rest.copy(); flex[:, 0] += rng.normal(0, 50, 300)
        profile = compute_calibration(rest, flex)
        self.assertEqual(profile.quality[0], "GOOD")
        self.assertIn(profile.quality[1], {"DEAD", "WEAK"})

    def test_signal_quality_rejects_nonfinite(self):
        profile = CalibrationProfile(np.zeros(2), np.ones(2), np.ones(2), np.ones(2)*5, np.ones(2)*5, np.ones(2)*5, ["GOOD"]*2, True, [])
        self.assertEqual(assess_signal_quality(np.array([[np.nan, 1]], dtype=np.float32), profile).state, "SIGNAL_POOR")

    def test_feature_count_and_compatibility(self):
        self.assertEqual(expected_feature_count(4), 70)
        artifact = {"model": FakeModel(), "class_names": ["REST", "UP"], "sample_rate": 500, "input_channels": 4}
        self.assertTrue(validate_model_artifact(artifact, 500, 4).compatible)
        self.assertFalse(validate_model_artifact(artifact, 250, 4).compatible)

    def test_temporal_and_release_gating(self):
        d = GestureDecisionEngine(vote_windows=3, consecutive_required=2, min_margin=.1)
        self.assertFalse(d.update("UP", .9, .2, True, 1).action_allowed)
        self.assertTrue(d.update("UP", .9, .2, True, 2).action_allowed)
        self.assertFalse(d.update("REST", .9, .2, True, 3).action_allowed)
        self.assertTrue(d.update("UP", .9, .2, True, 4).action_allowed is False)

    def test_ring_wrap_and_discontinuity_recovery(self):
        ring = SampleRingBuffer(2, 5)
        ring.append(np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float32))
        ring.append(np.array([[4, 40], [5, 50], [6, 60]], dtype=np.float32))
        np.testing.assert_array_equal(ring.latest(5), np.array([[2, 20], [3, 30], [4, 40], [5, 50], [6, 60]], dtype=np.float32))
        ring.append(np.array([[7, 70]], dtype=np.float32), discontinuity=True)
        self.assertFalse(ring.has_window(5))
        ring.append(np.array([[8, 80], [9, 90], [10, 100], [11, 110]], dtype=np.float32))
        self.assertTrue(ring.has_window(5))

    def test_optimized_feature_equivalence(self):
        rng = np.random.default_rng(123)
        window = rng.normal(size=(100, 4)).astype(np.float32)
        np.testing.assert_allclose(extract_window_features_optimized(window, 500),
                                   extract_window_features_legacy(window, 500), rtol=2e-5, atol=2e-4)


if __name__ == "__main__":
    unittest.main()

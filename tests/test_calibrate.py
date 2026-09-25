import io
import unittest
from contextlib import redirect_stderr

from tests.test_watch import PERSONAS, FakeCaller, target
from proof_of_edition.watch.calibrate import calibrate


class CalibrateTests(unittest.TestCase):
    def test_same_deployment_floor(self):
        targets = [target("same", "model-x")]
        with redirect_stderr(io.StringIO()):
            report = calibrate(targets, samples=6, permutations=200, caller=FakeCaller(PERSONAS), clock=lambda: 1_800_000_000)
        entry = report["targets"]["same"]
        self.assertEqual(entry["deterministic"]["exact_rate"], 1.0)
        self.assertTrue(entry["sampled"]["same_distribution"])
        self.assertEqual(entry["suggested_thresholds"]["min_exact_rate"], 0.85)
        self.assertEqual(entry["warnings"], [])

    def test_missing_key_is_reported(self):
        from proof_of_edition.watch.client import Target
        t = Target(name="needs-key", model="m", base_url="u", upstream_model="m", api_key_env="WATCH_TEST_MISSING_KEY")
        with redirect_stderr(io.StringIO()):
            report = calibrate([t], samples=2, permutations=10, caller=FakeCaller({}), clock=lambda: 1_800_000_000)
        self.assertIn("not set", report["targets"]["needs-key"]["skipped"])


if __name__ == "__main__":
    unittest.main()

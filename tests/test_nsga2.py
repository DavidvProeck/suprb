from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle as apply_shuffle
import unittest
import itertools
import numpy as np

from suprb import SupRB
from suprb import rule
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import ga
from suprb.optimizer.rule.nsga2 import NSGA2
from suprb.rule.matching import OrderedBound
from suprb.utils import check_random_state


class TestNSGA2Metrics(unittest.TestCase):
    """Test NSGA2 implementation and metric registry combinations"""

    def setup_test_example(self):
        n_samples = 500
        random_state = 42
        random_state_ = check_random_state(random_state)

        X = np.linspace(0, 20, num=n_samples)
        y = np.zeros(n_samples)
        y[X < 10] = np.sin(np.pi * X[X < 10] / 5) + 0.2 * np.cos(4 * np.pi * X[X < 10] / 5)
        y[X >= 10] = X[X >= 10] / 10
        y += random_state_.normal(scale=0.1, size=n_samples)

        X = X.reshape((-1, 1))
        X, y = apply_shuffle(X, y, random_state=random_state)

        self.X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
        self.y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    def setup_base_model(self, metrics_config):
        self.model = SupRB(
            rule_discovery=NSGA2(
                n_iter=5,
                mu=16,
                lmbda=32,
                metrics=metrics_config,
                random_state=42,
                n_jobs=1,
            ),
            solution_composition=ga.GeneticAlgorithm(
                n_iter=10,
                population_size=16,
                random_state=42,
                n_jobs=1,
            ),
            matching_type=OrderedBound(np.array([-1, 1])),
            n_iter=2,
            n_rules=4,
            verbose=0,
            logger=CombinedLogger([("stdout", StdoutLogger()), ("default", DefaultLogger())]),
        )

    def setup_metric_combinations(self):
        # All available string keys registered in NSGA2Objectives
        available_metrics = [
            "error",
            "volume",
            "support",
            "variance_reduction",
            "information_gain",
        ]

        # Generate combinations (e.g., pairs and triplets) to test multi-objective scaling
        self.metric_configs = []
        for r in range(1, len(available_metrics) + 1):
            for combo in itertools.combinations(available_metrics, r):
                self.metric_configs.append(list(combo))

    def setUp(self) -> None:
        self.setup_test_example()
        self.setup_metric_combinations()
        return super().setUp()

    def test_smoke_test(self):
        for metrics_config in self.metric_configs:
            print(f"\n\nChecking NSGA2 with metrics: {metrics_config}")

            try:
                self.setup_base_model(metrics_config)
                self.assertTrue(True)
                print("PASSED [Model Generation]\n")
            except Exception as e:
                self.fail(f"FAILED! Model generation with metrics {metrics_config}: {e}")

            try:
                self.model.fit(self.X, self.y)
                self.assertTrue(True)
                print("PASSED [Model fit]\n")
            except Exception as e:
                self.fail(f"FAILED! Model fit with metrics {metrics_config}: {e}")


if __name__ == "__main__":
    unittest.main()

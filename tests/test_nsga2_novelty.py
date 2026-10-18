from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle as apply_shuffle
import unittest
import numpy as np

from suprb import SupRB
from suprb import rule
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import ga
from suprb.optimizer.rule.nsga2.nsga2_novelty import NSGA2Novelty
from suprb.optimizer.rule.ns.novelty_calculation import NoveltyCalculation
from suprb.optimizer.rule.ns.archive import ArchiveNovel
from suprb.optimizer.rule.ns.novelty_search_type import NoveltySearchType
from suprb.rule.matching import OrderedBound
from suprb.utils import check_random_state


class TestNSGA2NoveltyPError(unittest.TestCase):
    """Test NSGA2Novelty implementation for mode 'P' with error metric"""

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

    def setup_base_model(self):
        self.model = SupRB(
            rule_discovery=NSGA2Novelty(
                n_iter=3,
                mu=16,
                lmbda=32,
                metrics=["error"],
                novelty_mode="P",
                novelty_calc=NoveltyCalculation(
                    novelty_search_type=NoveltySearchType(),
                    archive=ArchiveNovel(),
                    k_neighbor=10,
                ),
                min_experience=2,
                max_restarts=2,
                random_state=42,
                n_jobs=1,
            ),
            solution_composition=ga.GeneticAlgorithm(
                n_iter=5,
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

    def setUp(self) -> None:
        self.setup_test_example()
        return super().setUp()

    def test_smoke_test(self):
        print("\n\nChecking NSGA2Novelty with mode=P and metrics=['error']")

        try:
            self.setup_base_model()
            self.assertTrue(True)
            print("PASSED [Model Generation]\n")
        except Exception as e:
            self.fail(f"FAILED! Model generation: {e}")

        try:
            self.model.fit(self.X, self.y)
            self.assertTrue(True)
            print("PASSED [Model fit]\n")
        except Exception as e:
            self.fail(f"FAILED! Model fit: {e}")


if __name__ == "__main__":
    unittest.main()
import numpy as np
from typing import Optional, List, Callable

from suprb.rule import Rule, RuleInit
from suprb.utils import RandomState

from src.suprb.suprb.rule.initialization import MeanInit
from ..origin import SquaredError, RuleOriginGeneration
from ..mutation import RuleMutation, HalfnormIncrease
from ..constraint import CombinedConstraint, MinRange, Clip
from ..acceptance import Variance
from .. import RuleAcceptance, RuleConstraint

from .nsga2 import NSGA2

class NSGA2InfoGain(NSGA2):
    """
    NSGA2 with error and information gain as objectives.
    TODO: Better Description
    """
    def __init__(
        self,
        n_iter: int = 32,
        mu: int = 50,
        lmbda: int = 100,
        origin_generation: RuleOriginGeneration = SquaredError(),
        init: RuleInit = MeanInit(),
        mutation: RuleMutation = HalfnormIncrease(sigma=1.22),
        constraint: RuleConstraint = CombinedConstraint(MinRange(), Clip()),
        acceptance: RuleAcceptance = Variance(),
        random_state: int = None,
        n_jobs: int = 1,
        fitness_objs: Optional[List[Callable[[Rule], float]]] = None,
        fitness_objs_labels: Optional[List[str]] = None,
        profile: bool = False,
    ):
        super().__init__(
            n_iter=n_iter,
            mu=mu,
            lmbda=lmbda,
            origin_generation=origin_generation,
            init=init,
            mutation=mutation,
            constraint=constraint,
            acceptance=acceptance,
            random_state=random_state,
            n_jobs=n_jobs,
            fitness_objs=fitness_objs,
            fitness_objs_labels=fitness_objs_labels,
            profile=profile,
        )


    def _optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_rule: Rule,
        random_state: RandomState
    ):
        # Bind dataset for the IG objective
        self._X_ref = X
        self._y_ref = y
        self._H_y   = self._entropy(y)

        self.fitness_objs = list(self.fitness_objs or []) + [
            (lambda r, X_ref=X, y_ref=y, H_y=self._H_y: -self._information_gain(r.match(X_ref), y_ref, H_y)),
        ]
        self.fitness_objs_labels = list(self.fitness_objs_labels or []) + ["-IG"]

        return super()._optimize(X, y, initial_rule, random_state)


# ────────────────────────────────────────────────────────────────────
# Information Gain Helpers
# ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _entropy(y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        vals, cnt = np.unique(y, return_counts=True)
        p = cnt / cnt.sum()
        return float(-np.sum(p * np.log2(p + 1e-12)))


    @staticmethod
    def _information_gain(mask: np.ndarray, y: np.ndarray, H_y: float) -> float:
        if mask.size == 0:
            return 0.0
        p = float(mask.mean())
        if p == 0.0 or p == 1.0:
            return 0.0
        H_m = NSGA2InfoGain._entropy(y[mask])
        H_nm = NSGA2InfoGain._entropy(y[~mask])
        return H_y - (p * H_m + (1.0 - p) * H_nm)
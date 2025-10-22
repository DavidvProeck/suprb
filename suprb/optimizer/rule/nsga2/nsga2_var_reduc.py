import numpy as np
from typing import Optional, List, Callable

from suprb.rule import Rule, RuleInit
from suprb.utils import RandomState

from suprb.rule.initialization import MeanInit
from ..origin import SquaredError, RuleOriginGeneration
from ..mutation import RuleMutation, Normal
from ..constraint import CombinedConstraint, MinRange, Clip
from ..acceptance import Variance
from .. import RuleAcceptance, RuleConstraint

from .nsga2 import NSGA2


class NSGA2VarianceReduction(NSGA2):
    """
    NSGA-II variant using rule error and variance reduction (regression-style information gain)
    as optimization objectives.
    """

    def __init__(
        self,
        n_iter: int = 32,
        mu: int = 50,
        lmbda: int = 100,
        origin_generation: RuleOriginGeneration = SquaredError(),
        init: RuleInit = MeanInit(),
        mutation: RuleMutation = Normal(sigma=1.22),
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

    # ────────────────────────────────────────────────────────────────
    # Main optimization entry
    # ────────────────────────────────────────────────────────────────
    def _optimize(self, X: np.ndarray, y: np.ndarray, random_state: RandomState):
        self._var_y = np.var(y)

        self._varred_obj = lambda r, X_ref=X, y_ref=y, var_y=self._var_y: -self._variance_reduction(
            r.match(X_ref), y_ref, var_y
        )
        self._varred_label = "-Variance Reduction"

        return super()._optimize(X, y, random_state)

    # ────────────────────────────────────────────────────────────────
    # Variance Reduction Helpers
    # ────────────────────────────────────────────────────────────────
    @staticmethod
    def _variance_reduction(mask: np.ndarray, y: np.ndarray, var_y: float) -> float:
        """
        Variance-based analogue to information gain for regression.
        IG_var = Var(y) - [p * Var(y[mask]) + (1 - p) * Var(y[~mask])]
        """
        if mask.size == 0:
            return 0.0
        p = float(mask.mean())
        if p == 0.0 or p == 1.0:
            return 0.0

        var_match = np.var(y[mask]) if np.any(mask) else 0.0
        var_not_match = np.var(y[~mask]) if np.any(~mask) else 0.0
        return var_y - (p * var_match + (1.0 - p) * var_not_match)

    # ────────────────────────────────────────────────────────────────
    # Helper functions
    # ────────────────────────────────────────────────────────────────
    def _fitness_objs_runtime(self) -> List[Callable[[Rule], float]]:
        return list(self.fitness_objs) + [self._varred_obj]

    def _fitness_labels_runtime(self) -> List[str]:
        return list(self.fitness_objs_labels) + [self._varred_label]

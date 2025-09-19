import numpy as np

from suprb.rule import Rule

from suprb.optimizer.rule import RuleConstraint

from pymoo.core.problem import Problem


class RuleOptimizationProblem(Problem):
    def __init__(self, rule: Rule, X: np.ndarray, y: np.ndarray, constraint: RuleConstraint):
        self.rule_template = rule
        self.X = X
        self.y = y
        self.param_len = len(rule.get_param_vector())
        self.constraint = constraint

        xl = np.full(self.param_len, -1.0)
        xu = np.full(self.param_len, 1.0)

        super().__init__(
            n_var=self.param_len,
            n_obj=2,
            n_constr=0,
            xl=xl,
            xu=xu,
            elementwise_evaluation=False
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for x in X:
            # 1) build & inject the candidate vector
            rule = self.rule_template.clone().set_param_vector(x)

            # 2) fit to produce new match.bounds, error_, experience_, etc.
            rule = rule.fit(self.X, self.y)

            if not rule.is_fitted_ or rule.experience_ == 0:
                F.append([np.inf, np.inf])
            else:
                # 3) now apply your constraint to the *fitted* bounds
                rule = self.constraint(rule)
                # 4) record error & NEGATIVE volume
                F.append([rule.error_, -rule.volume_])

        out["F"] = np.array(F)

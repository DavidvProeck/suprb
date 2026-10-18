import numpy as np
from typing import List, Callable, Tuple, Optional


class NSGA2Objectives:
    """
    Helper class for defining simple rule fitness objectives for NSGA2.
    Can only be used for objectives that do not change the structure of the algorithm.
    """

    @staticmethod
    def get_dynamic_objectives(
        X: np.ndarray,
        y: np.ndarray,
        fitness_objs: Tuple[Callable, ...],
        fitness_labels: Tuple[str, ...],
        metrics: Optional[List[str]] = None,
    ) -> Tuple[List[Callable], List[str]]:
        objs = list(fitness_objs)
        labels = list(fitness_labels)

        if not metrics:
            return objs, labels

        var_y = np.var(y) if "variance_reduction" in metrics else None
        H_y = NSGA2Objectives.entropy(y) if "information_gain" in metrics else None

        registry = {
            "error": (lambda r: r.error_, "Error"),
            "volume": (lambda r: -r.volume_, "-Volume"),
            "support": (lambda r: -np.count_nonzero(r.match_set_), "-Support"),
            "variance_reduction": (
                lambda r, v=var_y: -NSGA2Objectives.variance_reduction(r.match(X), y, v),
                "-Variance Reduction",
            ),
            "information_gain": (
                lambda r, h=H_y: -NSGA2Objectives.information_gain(r.match(X), y, h),
                "-Information Gain",
            ),
        }

        for name in metrics:
            if name in registry:
                obj_func, label = registry[name]
                objs.append(obj_func)
                labels.append(label)
            else:
                raise ValueError(f"Unknown metric objective: '{name}'. Available: {list(registry.keys())}")

        return objs, labels

    @staticmethod
    def variance_reduction(mask: np.ndarray, y: np.ndarray, var_y: float) -> float:
        if mask.size == 0 or mask.mean() in [0.0, 1.0]:
            return 0.0
        p = float(mask.mean())
        var_match = np.var(y[mask]) if np.any(mask) else 0.0
        var_not_match = np.var(y[~mask]) if np.any(~mask) else 0.0
        return var_y - (p * var_match + (1.0 * p) * var_not_match)

    @staticmethod
    def entropy(y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        _, cnt = np.unique(y, return_counts=True)
        p = cnt / cnt.sum()
        return float(-np.sum(p * np.log2(p + 1e-12)))

    @staticmethod
    def information_gain(mask: np.ndarray, y: np.ndarray, H_y: float) -> float:
        if mask.size == 0 or mask.mean() in [0.0, 1.0]:
            return 0.0
        p = float(mask.mean())
        H_m = NSGA2Objectives.entropy(y[mask])
        H_nm = NSGA2Objectives.entropy(y[~mask])
        return H_y - (p * H_m + (1.0 * p) * H_nm)

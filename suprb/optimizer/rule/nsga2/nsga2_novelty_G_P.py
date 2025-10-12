import copy
import warnings
from collections import deque

import numpy as np
from typing import Literal, Optional, List, Callable, Set, Deque
from joblib import Parallel, delayed

from suprb import Solution
from suprb.rule import Rule, RuleInit
from suprb.rule.initialization import MeanInit
from suprb.optimizer.rule.nsga2.nsga2 import NSGA2
from suprb.optimizer.rule.ns.novelty_calculation import NoveltyCalculation
from suprb.optimizer.rule.ns.archive import ArchiveNovel
from suprb.optimizer.rule.ns.novelty_search_type import NoveltySearchType
from suprb.solution.fitness import PseudoBIC
from suprb.solution.mixing_model import ErrorExperienceHeuristic
from suprb.utils import RandomState
from .nsga2_helpers import visualize_pareto_front
from ..origin import SquaredError, RuleOriginGeneration
from ..mutation import RuleMutation, HalfnormIncrease, Normal
from ..constraint import CombinedConstraint, MinRange, Clip
from ..acceptance import Variance
from .. import RuleAcceptance, RuleConstraint

import cProfile
import pstats


class NSGA2Novelty_G_P(NSGA2):
    """
    NSGA-II variant that adds novelty as an objective and supports restart logic:
    If a run's Pareto front only contains trivial rules (e.g., experience == 1),
    the algorithm restarts and accumulates only 'useful' rules until `mu` useful
    rules are collected or a restart cap is hit.
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
            novelty_calc: NoveltyCalculation = NoveltyCalculation(
                novelty_search_type=NoveltySearchType(),
                archive=ArchiveNovel(),
                k_neighbor=15,
            ),
            novelty_mode: Literal["G", "P"] = "P",
            profile: bool = False,
            min_experience: int = 2,
            max_restarts: int = 5,
            keep_archive_across_restarts: bool = True,
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
        )
        self.novelty_calc = novelty_calc
        self.novelty_mode = novelty_mode
        self.profile = profile

        self.min_experience = min_experience
        self.max_restarts = max_restarts
        self.keep_archive_across_restarts = keep_archive_across_restarts

        self._novelty_obj = lambda r: -getattr(r, "novelty_score_", np.inf)
        self._novelty_label = "-Novelty"


        self.last_front_: List[Rule] = []

        self.archive_maxlen = 1000
        self.local_pool_: Deque[Rule] = deque(maxlen=self.archive_maxlen)
        self._archive_seen_ids: Set[int] = set()


    # ────────────────────────────────────────────────────────────────
    # Novelty scoring
    # ────────────────────────────────────────────────────────────────
    # def _score_novelty(
    #     self,
    #     rules: List[Rule],
    #     cohort: Optional[List[Rule]] = None,
    #     force: bool = False,
    # ) -> None:
    #     #TODO: Redo the pseudocode in paper
    #     if not rules:
    #         return
    #
    #     to_score = list(rules) if force else [r for r in rules if not hasattr(r, "novelty_score_")]
    #     if not to_score:
    #         return
    #
    #     if self.novelty_mode == "G":
    #         ref = []
    #         seen = set()
    #
    #         def _extend_unique(src):
    #             for r in src:
    #                 rid = id(r)
    #                 if rid not in seen:
    #                     r_clone = copy.deepcopy(r)
    #                     ref.append(r_clone)
    #                     seen.add(id(r_clone))
    #
    #         _extend_unique(self.pool_)
    #         _extend_unique(self.local_pool_)
    #
    #         if cohort and self.last_front_:
    #             _extend_unique(self.last_front_)
    #         elif cohort:
    #             _extend_unique(cohort)
    #     else:  # "P" — use only the current cohort (or the given rules)
    #         ref = list(cohort) if cohort else list(rules)
    #
    #     ref = self._cap_list(ref) # Cap archive size
    #
    #     self.novelty_calc.archive.archive = ref
    #     _ = self.novelty_calc(to_score)
    #
    #     if self.novelty_mode == "G":
    #         seen_local = set(map(id, self.local_pool_))
    #         self.local_pool_.extend([r for r in to_score if id(r) not in seen_local])

    def _score_novelty(self,
        rules: List[Rule],
        cohort: Optional[List[Rule]] = None,
        force: bool = False,
    ) -> None:
        if not rules:
            return

        to_score = list(rules) if force else [r for r in rules if not hasattr(r, "novelty_score_")]
        if not to_score:
            return
        if self.novelty_mode == "G":
            for r in to_score:
                rid = id(r)
                if rid not in self._archive_seen_ids:
                    self.local_pool_.append(copy.deepcopy(r))
                    self._archive_seen_ids.add(rid)

            ref = list(self.local_pool_)

            if cohort and self.last_front_:
                extra = self.last_front_
            else:
                extra = cohort

            if extra:
                EXTRA_MAX = 256
                extra_slice = extra[-EXTRA_MAX:] if len(extra) > EXTRA_MAX else extra
                ref.extend(copy.deepcopy(extra_slice))
        #novelty_mode = "P"
        else:
            ref = list(cohort) if cohort else list(rules)

        ref = self._cap_list(ref)
        self.novelty_calc.archive.archive = ref
        _ = self.novelty_calc(to_score)


    # ────────────────────────────────────────────────────────────────
    # Helpers for restart logic
    # ────────────────────────────────────────────────────────────────
    def _is_useful(self, r: Rule) -> bool:
        """Define 'useful' rules here."""
        return getattr(r, "experience_", 0) >= self.min_experience

    def _unique_extend(self, base: List[Rule], new_rules: List[Rule]) -> None:
        seen = set(map(id, base))
        for r in new_rules:
            if id(r) not in seen:
                base.append(r)
                seen.add(id(r))



    # ────────────────────────────────────────────────────────────────
    # One full NSGA-II run: returns a Pareto front
    # ────────────────────────────────────────────────────────────────
    def _run_once(
            self,
            X: np.ndarray,
            y: np.ndarray,
            random_state: RandomState,
            clear_pool: bool,
    ) -> Optional[List[Rule]]:

        if clear_pool:
            self.local_pool_.clear()
            self._archive_seen_ids.clear()
            self.last_front_ = []

        origins = self.origin_generation(
            n_rules=self.mu,
            X=X,
            y=y,
            pool=self.pool_,
            elitist=self.elitist_,  # will be non-None if your init.model trained successfully
            random_state=random_state,
        )

        profiler = cProfile.Profile() if self.profile else None
        if profiler:
            profiler.enable()


        population = Parallel(n_jobs=self.n_jobs)(
            delayed(self._init_valid_origin)(origin, X, y, random_state) for origin in origins
        )
        population = [p for p in population if p is not None]
        if not population:
            if profiler:
                profiler.disable()
            return None

        self._score_novelty(population, cohort=population, force=True)

        # main loop
        for _ in range(self.n_iter):
            parents = random_state.choice(population, size=self.lmbda, replace=True)
            children = Parallel(n_jobs=self.n_jobs)(
                delayed(self._generate_valid_child)(parent, X, y, random_state) for parent in parents
            )
            children = [c for c in children if c is not None]
            if not children:
                continue

            cohort = population + children
            self._score_novelty(children, cohort=cohort, force=False)
            self._score_novelty(population, cohort=cohort, force=True)

            population_combined = population + children

            pareto_fronts = self._fast_nondominated_sort(population_combined)
            self.last_front_ = pareto_fronts[0]

            population = self._build_next_population(pareto_fronts)

        pareto_front = pareto_fronts[0] if pareto_fronts else []

        if profiler:
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats("cumtime")
            stats.print_stats(20)

        return pareto_front

    # ────────────────────────────────────────────────────────────────
    # Running until `mu` useful rules are collected or max_restarts is hit
    # ────────────────────────────────────────────────────────────────
    def _optimize(
            self,
            X: np.ndarray,
            y: np.ndarray,
            random_state: RandomState,
    ) -> Optional[List[Rule]]:

        useful_rules: List[Rule] = []
        restarts = 0

        while len(useful_rules) < self.mu and restarts <= self.max_restarts:
            pareto_front = self._run_once(
                X=X,
                y=y,
                random_state=random_state,
                clear_pool=(not self.keep_archive_across_restarts),
            )

            if not pareto_front:
                restarts += 1
                continue

            useful_from_front = [r for r in pareto_front if self._is_useful(r)]
            self._unique_extend(useful_rules, useful_from_front)

            only_trivial = all(getattr(r, "experience_", 0) < self.min_experience for r in pareto_front)

            if only_trivial and len(useful_rules) < self.mu:
                restarts += 1
                continue

            if len(useful_rules) < self.mu:
                restarts += 1
                continue

        # Truncate to mu if we got more (optional)
        if useful_rules:
            useful_rules = useful_rules[: self.mu]

        # to_visualize = useful_rules if useful_rules else (pareto_front or [])
        # if to_visualize:
        #     visualize_pareto_front(self, to_visualize) #TODO: Fix function

        print(f"Iterations needed to generate mu useful rules: {restarts + 1}")
        return useful_rules if useful_rules else (pareto_front or None)

    # ────────────────────────────────────────────────────────────────
    # Helper functions
    # ────────────────────────────────────────────────────────────────
    def _fitness_objs_runtime(self) -> List[Callable[[Rule], float]]:
        return list(self.fitness_objs) + [self._novelty_obj]

    def _fitness_labels_runtime(self) -> List[str]:
        return list(self.fitness_objs_labels) + [self._novelty_label]

    def _cap_list(self, seq: List[Rule]) -> List[Rule]:
        if len(seq) <= self.archive_maxlen:
            return seq

        return seq[-self.archive_maxlen:]

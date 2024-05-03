import numpy as np
from box import Box
from more_itertools import grouper
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.problem import LoopedElementwiseEvaluation
from pymoo.optimize import minimize
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from toolz.curried import pipe

from mlutils.scikit.competence_region_ensemble import SimpleCompetenceRegionEnsembleV2

from optimalcentroids.lib import nn_wrapper, find_closeset_val

DEFAULT_PARAMS = {
    'max_depth': 5,
    'n_trees': 5,
    "cv": 2,
    "cv_repeats": 5,
    "n_jobs": -1,
    'n_gen': 20,
    'pop_size': 20,
    'debug': False,
}


def create_estimator(centroids, depths):
    activated_trees_indices = np.nonzero(depths)[0]

    active_centroids = centroids[activated_trees_indices]
    active_depths = depths[activated_trees_indices]

    space_classifier = NearestNeighbors()
    space_classifier.fit(active_centroids)

    model = SimpleCompetenceRegionEnsembleV2(
        nn_wrapper(space_classifier),
        {label: DecisionTreeClassifier(max_depth=depth, random_state=42) for label, depth in enumerate(active_depths)}
    )

    return model


class OptimalCentroidPositionProblem(ElementwiseProblem):

    def __init__(self, n_trees, x_train, y_train, max_tree_depth, params, **kwargs):
        n_dim = x_train.shape[1]

        super().__init__(
            n_var=n_trees * n_dim + n_trees,  # each centroid * number of features + depths
            n_obj=1,  # single metric
            n_constr=0,
            xl=list(np.min(x_train, axis=0)) * n_trees + n_trees * [-0.5],
            xu=list(np.max(x_train, axis=0)) * n_trees + n_trees * [max_tree_depth + 0.5],
            **kwargs
        )

        self.params = params
        self.n_trees = n_trees
        self.x_train = x_train
        self.y_train = y_train
        self.n_dim = n_dim
        self.max_tree_depth = max_tree_depth

    def _evaluate(self, individual, out, *args, **kwargs):
        n_coordinates_in_individual = self.n_dim * self.n_trees
        centroid_coordinates = individual[:n_coordinates_in_individual]

        individual_as_centroids = pipe(
            centroid_coordinates,
            lambda x: grouper(x, self.n_dim),
            list,
            np.array,
            np.nan_to_num
        )

        tree_depths_continous = individual[-self.n_trees:]
        possible_tree_depths = list(range(self.max_tree_depth + 1))

        tree_depths = np.array([find_closeset_val(possible_tree_depths, td) for td in tree_depths_continous])

        if np.all(tree_depths == 0):
            out["F"] = 1
        else:
            model = create_estimator(individual_as_centroids, tree_depths)

            skf = RepeatedKFold(n_splits=self.params['cv'], n_repeats=self.params['cv_repeats'], random_state=42)
            scores = cross_validate(model, self.x_train, self.y_train, n_jobs=self.params['n_jobs'], scoring='accuracy',
                                    cv=skf)

            if self.params.debug:
                print(f"Depths = {tree_depths}, acc = {scores['test_score'].mean()}")

            out["F"] = 1 - scores['test_score'].mean()


def run(x_train, y_train, n_trees, max_tree_depth, pop_size, n_gen, pymoo_elementwise_runner=LoopedElementwiseEvaluation()):
    params = Box({**DEFAULT_PARAMS, **{
        'x_train': x_train,
        'n_trees': n_trees,
        'pop_size': pop_size,
        'n_gen': n_gen,
        'max_depth': max_tree_depth
    }})

    problem = OptimalCentroidPositionProblem(n_trees, x_train, y_train, max_tree_depth, params, elementwise_runner=pymoo_elementwise_runner)

    res = minimize(problem,
                   GA(
                       pop_size=pop_size,
                       verbose=True,
                       seed=42,
                       eliminate_duplicates=True
                   ),
                   ("n_gen", 10),
                   verbose=True,
                   save_history=True,
                   seed=42)

    if res.X.ndim == 1:
        pareto_front = [res.X]
    else:
        pareto_front = res.X

    n_dim = problem.n_dim
    models = []
    
    for individual in pareto_front:
        n_coordinates_in_individual = n_dim * n_trees
        centroid_coordinates = individual[:n_coordinates_in_individual]

        individual_as_centroids = pipe(
            centroid_coordinates,
            lambda x: grouper(x, n_dim),
            list,
            np.array,
            np.nan_to_num
        )

        tree_depths_continous = individual[-n_trees:]
        possible_tree_depths = list(range(params.max_depth + 1))

        tree_depths = np.array([find_closeset_val(possible_tree_depths, td) for td in tree_depths_continous])

        if params.debug:
            print(f"Depths = {tree_depths}")

        if np.all(tree_depths==0):
            continue
        else:
            model = create_estimator(individual_as_centroids, tree_depths)
            model.fit(x_train, y_train)

            models.append(model)

    return models

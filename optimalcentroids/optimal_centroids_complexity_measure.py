import numba
import numpy as np
from box import Box
from more_itertools import grouper
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.problem import LoopedElementwiseEvaluation
from pymoo.optimize import minimize
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.svm import LinearSVC
from loguru import logger as log
from sklearn.neighbors import NearestNeighbors
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from toolz.curried import pipe
import problexity as px
from mlutils.utils import compute_metric_ovo
from .lib import list_with_repeated_elements, nn_wrapper, find_closeset_val

from mlutils.scikit.competence_region_ensemble import SimpleCompetenceRegionEnsembleV2


def create_space_classifier(centroids):

    space_classifier = NearestNeighbors(n_neighbors=len(centroids))
    space_classifier.fit(centroids)

    return space_classifier


class OptimalCentroidComplexityMeasurePositionProblem(ElementwiseProblem):

    def __init__(self, n_clfs, x_train, y_train, complexity_metric, **kwargs):
        n_dim = x_train.shape[1]

        super().__init__(
            n_var=n_clfs * n_dim,  # each centroid * number of features
            n_obj=1,  # single metric
            n_constr=0,
            xl=list(np.min(x_train, axis=0)) * n_clfs,
            xu=list(np.max(x_train, axis=0)) * n_clfs,
            **kwargs
        )

        self.n_clfs = n_clfs
        self.x_train = x_train
        self.y_train = y_train
        self.n_dim = n_dim
        self.complexity_metric = complexity_metric

    def _evaluate(self, individual, out, *args, **kwargs):
        n_coordinates_in_individual = self.n_dim * self.n_clfs
        centroid_coordinates = individual[:n_coordinates_in_individual]

        individual_as_centroids = pipe(
            centroid_coordinates,
            lambda x: grouper(x, self.n_dim),
            list,
            np.array,
            np.nan_to_num
        )

        space_classifier = create_space_classifier(individual_as_centroids)

        centroid_assignments = space_classifier.kneighbors(self.x_train, n_neighbors=1, return_distance=False)

        complexities = []
        for clz in np.unique(centroid_assignments):
            x_in_centroid = self.x_train[np.where(centroid_assignments == clz)[0]]
            y_in_centroid = self.y_train[np.where(centroid_assignments == clz)[0]]

            try:
                complexities.append(np.mean(compute_metric_ovo(x_in_centroid, y_in_centroid, self.complexity_metric)))
            except Exception as e:
                log.warning(e)
                out["F"] = 1
                return

        out["F"] = np.array(complexities).mean()

def run(x_train, y_train, n_clf, n_gen = 10, n_pop = 10, model = LinearSVC(random_state=42), complexity_metric = px.l1, pymoo_elementwise_runner=LoopedElementwiseEvaluation()):
    problem = OptimalCentroidComplexityMeasurePositionProblem(
        n_clfs=n_clf,
        complexity_metric=complexity_metric,
        x_train=x_train,
        y_train=y_train,
        elementwise_runner=pymoo_elementwise_runner)

    res = minimize(problem,
                   GA(
                       pop_size=n_pop,
                       verbose=True,
                       seed=42,
                       eliminate_duplicates=True
                   ),
                   ("n_gen", n_gen),
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
        n_coordinates_in_individual = n_dim * n_clf
        centroid_coordinates = individual[:n_coordinates_in_individual]

        individual_as_centroids = pipe(
            centroid_coordinates,
            lambda x: grouper(x, n_dim),
            list,
            np.array,
            np.nan_to_num
        )

        space_classifier = create_space_classifier(individual_as_centroids)

        final_model = SimpleCompetenceRegionEnsembleV2(
            nn_wrapper(space_classifier),
            {label: clone(model) for label in range(n_clf)}
        )
        final_model.fit(x_train, y_train)

        models.append(final_model)

    return models

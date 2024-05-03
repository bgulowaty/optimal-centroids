
import numpy as np
from mlutils.scikit.competence_region_ensemble import SimpleCompetenceRegionEnsembleV2
from more_itertools import grouper
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.neighbors import NearestNeighbors
from toolz.curried import pipe

from optimalcentroids.lib import nn_wrapper, individual_to_centroid


def create_estimator(centroids, rf, x):

    space_classifier = NearestNeighbors()
    space_classifier.fit(centroids)
    wrapped_space_classifier = nn_wrapper(space_classifier)

    space_preds = wrapped_space_classifier.predict(x)

    best_tree_by_centroid = {}
    for idx, centroid in enumerate(centroids):
        x_in_centr = x[space_preds == idx]
        rf_preds = rf.predict(x_in_centr)
        for tree in rf.estimators_:
            tree_preds = tree.predict(x_in_centr)
            depth = tree.get_depth()
            score = 1/depth + accuracy_score(rf_preds, tree_preds)

            if not idx in best_tree_by_centroid.keys():
                best_tree_by_centroid[idx] = {
                    "tree": tree,
                    "score": score
                }
            elif score > best_tree_by_centroid[idx]["score"]:
                best_tree_by_centroid[idx] = {
                    "tree": tree,
                    "score": score
                }

    model = SimpleCompetenceRegionEnsembleV2(
        wrapped_space_classifier,
        {label: tree_details["tree"] for label, tree_details in best_tree_by_centroid.items()},
        dont_fit=True
    )

    return model

def create_space_classifier(centroids):

    space_classifier = NearestNeighbors()
    space_classifier.fit(centroids)

    return space_classifier

class OptimalCentroidExplainer(ElementwiseProblem):
    def __init__(self, rf, n_clusters, x_train, y_train, *args, **kwargs):
        n_dim = x_train.shape[1]

        super().__init__(
            n_var=n_clusters * n_dim,  # each centroid * number of features
            n_obj=2,
            n_constr=0,
            xl=list(np.min(x_train, axis=0)) * n_clusters,
            xu=list(np.max(x_train, axis=0)) * n_clusters,
            *args,
            **kwargs
        )

        self.n_clusters = n_clusters
        self.x_train = x_train
        self.y_train = y_train
        self.n_dim = n_dim
        self.rf = rf


    def build_model(self, individual):
        n_coordinates_in_individual = self.n_dim * self.n_clusters
        centroid_coordinates = individual[:n_coordinates_in_individual]

        individual_as_centroids = individual_to_centroid(centroid_coordinates, self.n_dim)

        return create_estimator(individual_as_centroids, self.rf, self.x_train)


    def _evaluate(self, individual, out, *args, **kwargs):
        n_coordinates_in_individual = self.n_dim * self.n_clusters
        centroid_coordinates = individual[:n_coordinates_in_individual]

        individual_as_centroids = individual_to_centroid(centroid_coordinates, self.n_dim)

        try:
            model = create_estimator(individual_as_centroids, self.rf, self.x_train)
        except Exception as e:
            print(e)
            out["F"] = [1, 9999]
            return
        skf = RepeatedKFold(n_splits=3, n_repeats=3, random_state=42)
        scores = cross_validate(model, self.x_train, self.rf.predict(self.x_train), n_jobs=1, scoring='accuracy', cv=skf, error_score='raise')
        acc = scores['test_score'].mean()
        mean_depth = np.average([tree.get_depth() for tree in model.clf_by_label.values()])

        print(f"Acc = {acc}, depth = {mean_depth}")

        out["F"] = [1 - acc, mean_depth]


class OptimalCentroidExplainerWithTreeSelection(ElementwiseProblem):
    def __init__(self, rf, n_clusters, x_train, y_train, *args, **kwargs):
        n_dim = x_train.shape[1]

        super().__init__(
            n_var=n_clusters * n_dim + n_clusters,
            n_obj=2,
            n_constr=0,
            xl=list(np.min(x_train, axis=0)) * n_clusters + n_clusters * [0],
            xu=list(np.max(x_train, axis=0)) * n_clusters + n_clusters * [len(rf.estimators_)],
            *args,
            **kwargs
        )

        self.n_clusters = n_clusters
        self.x_train = x_train
        self.y_train = y_train
        self.n_dim = n_dim
        self.rf = rf

    def build_model(self, individual):
        n_coordinates_in_individual = self.n_dim * self.n_clusters
        centroid_coordinates = individual[:n_coordinates_in_individual]
        selected_trees = individual[n_coordinates_in_individual:]
        individual_as_centroids = individual_to_centroid(centroid_coordinates, self.n_dim)

        return create_estimator_tree(individual_as_centroids, self.rf, selected_trees)

    def _evaluate(self, individual, out, *args, **kwargs):
        n_coordinates_in_individual = self.n_dim * self.n_clusters
        centroid_coordinates = individual[:n_coordinates_in_individual]
        selected_trees = individual[n_coordinates_in_individual:]

        individual_as_centroids = pipe(
            centroid_coordinates,
            lambda x: grouper(x, self.n_dim),
            list,
            np.array,
            np.nan_to_num
        )

        try:
            model = create_estimator_tree(individual_as_centroids, self.rf, selected_trees)
        except Exception as e:
            print(e)
            out["F"] = [1, 9999]
            return

        skf = RepeatedKFold(n_splits=3, n_repeats=3, random_state=42)
        scores = cross_validate(model, self.x_train, self.rf.predict(self.x_train), scoring='accuracy',
                                cv=skf)
        acc = scores['test_score'].mean()
        mean_depth = np.average([tree.get_depth() for tree in model.clf_by_label.values()])

        print(f"Acc = {acc}, depth = {mean_depth}")

        out["F"] = [1 - acc, mean_depth]


def create_estimator_tree(centroids, rf, trees):

    space_classifier = NearestNeighbors()
    space_classifier.fit(centroids)
    wrapped_space_classifier = nn_wrapper(space_classifier)

    tree_by_centroid = {
        idx: rf.estimators_[int(tree_idx)] for idx, tree_idx in enumerate(trees)
    }

    model = SimpleCompetenceRegionEnsembleV2(
        wrapped_space_classifier,
        tree_by_centroid,
        dont_fit=True
    )

    return model

def run(rf, n_clf, X, y, pop_size=10, n_gen=10):
    problem = OptimalCentroidExplainer(rf, n_clf, X, y)

    res = minimize(problem,
                   NSGA2(
                       pop_size=pop_size,
                       verbose=True,
                   ),
                   ("n_gen", n_gen),
                   verbose=True,
                   save_history=True,
                   seed=42)
    min_complexity_idx = np.argmin(res.F[:, 1], axis=0)
    max_acc_idx = np.argmin(res.F[:, 0], axis=0)

    return {
        "min_complexity_model": problem.build_model(res.X[min_complexity_idx]),
        "max_accuracy_model": problem.build_model(res.X[max_acc_idx])
    }

def run_tree(rf, n_clf, X, y, pop_size=10, n_gen=10):
    problem = OptimalCentroidExplainerWithTreeSelection(rf, n_clf, X, y)

    res = minimize(problem,
                   NSGA2(
                       pop_size=pop_size,
                       verbose=True,
                   ),
                   ("n_gen", n_gen),
                   verbose=True,
                   save_history=True,
                   seed=42)
    min_complexity_idx = np.argmin(res.F[:, 1], axis=0)
    max_acc_idx = np.argmin(res.F[:, 0], axis=0)

    return {
        "min_complexity_model": problem.build_model(res.X[min_complexity_idx]),
        "max_accuracy_model": problem.build_model(res.X[max_acc_idx])
    }



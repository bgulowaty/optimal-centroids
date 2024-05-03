import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

from optimalcentroids.optimal_centroids import run
from optimalcentroids.optimal_centroids_complexity_measure import run as run_complexity
from optimalcentroids.optimal_centroids_explainer import run as run_explainer, run_tree as run_explainer_tree


def test_runs_for_breast_cancer_data():
    # given
    train_data = pd.read_csv('../breast-train-0-s1.csv')
    x_train = train_data.drop('TARGET', axis=1).values
    y_train = train_data['TARGET'].values

    # expect
    try:
        run(x_train, y_train, 5, 5, 5, 5)
    except Exception as e:
        pytest.fail(f"raised an exception {e}")


def test_runs_for_complexity_measure_breast_cancer_data():
    # given
    train_data = pd.read_csv('../breast-train-0-s1.csv')
    x_train = train_data.drop('TARGET', axis=1).values
    y_train = train_data['TARGET'].values

    # expect
    try:
        models = run_complexity(x_train, y_train, 5, 10, 10)
    except Exception as e:
        pytest.fail(f"raised an exception {e}")


def test_runs_for_explainer_breast_cancer_data():
    # given
    train_data = pd.read_csv('../breast-train-0-s1.csv')
    X = train_data.drop('TARGET', axis=1).values
    y = train_data['TARGET'].values

    sss = StratifiedShuffleSplit(n_splits=1)
    rf = RandomForestClassifier(random_state=42)
    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
        rf.fit(X[train_index], y[train_index])
        try:
            models = run_explainer(rf, 5, X[test_index], y[test_index], 5, 5)
        except Exception as e:
            pytest.fail(f"raised an exception {e}")


def test_runs_for_tree_explainer_breast_cancer_data():
    # given
    train_data = pd.read_csv('../breast-train-0-s1.csv')
    X = train_data.drop('TARGET', axis=1).values
    y = train_data['TARGET'].values

    sss = StratifiedShuffleSplit(n_splits=1)
    rf = RandomForestClassifier(random_state=42)
    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
        rf.fit(X[train_index], y[train_index])
        try:
            models = run_explainer_tree(rf, 5, X[test_index], y[test_index], 5, 5)
        except Exception as e:
            pytest.fail(f"raised an exception {e}")

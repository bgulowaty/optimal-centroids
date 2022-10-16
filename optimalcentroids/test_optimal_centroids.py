import pandas as pd
import pytest

from .optimal_centroids import run


def test_runs_for_breast_cancer_data():
    # given
    train_data = pd.read_csv('breast-train-0-s1.csv')
    x_train = train_data.drop('TARGET', axis=1).values
    y_train = train_data['TARGET'].values

    # expect
    try:
        run(x_train, y_train, 5, 5, 5, 5)
    except Exception as e:
        pytest.fail(f"'sum_x_y' raised an exception {e}")

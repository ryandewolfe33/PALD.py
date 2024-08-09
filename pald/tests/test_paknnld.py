import pytest
import numpy as np

from sklearn.utils.estimator_checks import check_estimator

from pald import PAKNNLD


@pytest.fixture
def data():
    # Data is copied from exdata1 in R package
    X = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 1.28],
            [-1.0, -1.28],
            [0.0, 0.0],
            [1.4, -0.9],
            [1.4, 0.0],
            [1.4, 0.9],
            [0.5, 3.0],
        ]
    )
    yield X


def test_paknnld_internals(data):
    paknnld = PAKNNLD()
    paknnld.fit(data)

    assert hasattr(paknnld, "is_fitted_")
    np.testing.assert_equal(paknnld.labels_, [0, 2, 0, 0, 1, 1, 1, 2])


def test_paknnld_is_sklearn_estimator():
    paknnld = PAKNNLD()
    check_estimator(paknnld)

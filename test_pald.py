import pytest
import numpy as np
from PALD import PALD
import networkx as nx

@pytest.fixture
def setup_data():
    # Read test data from ex1data file and make dissimlarity matrix
    x = np.genfromtxt("exdata1.csv", delimiter=",")
    def distance(v1, v2):
        return np.sqrt(np.sum((v1 - v2) ** 2))  
    D = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
                d = distance(x[i, :], x[j, :])
                D[i,j] = D[j,i] = d

    yield D  # Provide the data to the test


def test_fit_ex1(setup_data):
    p = PALD()
    p.fit(setup_data)
    
    # Correct answer caluclated using the original pald R package.
    answer = np.array([
        [0.16581633, 0.11989796, 0.11989796, 0.01785714, 0.01785714, 0.00000000, 0.00000000, 0.00000000],
        [0.13061224, 0.17653061, 0.01785714, 0.11989796, 0.00000000, 0.00000000, 0.02040816, 0.07653061],
        [0.11870748, 0.02040816, 0.16462585, 0.10544218, 0.03826531, 0.01785714, 0.00000000, 0.00000000],
        [0.01785714, 0.10425170, 0.10425170, 0.17874150, 0.06207483, 0.09540816, 0.06207483, 0.00000000],
        [0.01785714, 0.00000000, 0.03571429, 0.05612245, 0.18707483, 0.12159864, 0.07993197, 0.00000000],
        [0.00000000, 0.00000000, 0.02040816, 0.07653061, 0.15986395, 0.20748299, 0.15986395, 0.01785714],
        [0.00000000, 0.01785714, 0.00000000, 0.05952381, 0.07993197, 0.12159864, 0.18707483, 0.03571429],
        [0.00000000, 0.07993197, 0.00000000, 0.00000000, 0.00000000, 0.01785714, 0.03571429, 0.15136054],
    ])

    np.testing.assert_allclose(p.pald, answer, rtol=1e-5, atol=1e-5)


def test_predict_ex1(setup_data):
    pald = np.array([
        [0.16581633, 0.11989796, 0.11989796, 0.01785714, 0.01785714, 0,          0,          0         ],
        [0.13061224, 0.17653061, 0.01785714, 0.11989796, 0,          0,          0.02040816, 0.07653061],
        [0.11870748, 0.02040816, 0.16462585, 0.10544218, 0.03826531, 0.01785714, 0,          0         ],
        [0.01785714, 0.1042517,  0.1042517,  0.1787415,  0.06207483, 0.09540816, 0.06207483, 0         ],
        [0.01785714, 0,          0.03571429, 0.05612245, 0.18707483, 0.12159864, 0.07993197, 0         ],
        [0,          0,          0.02040816, 0.07653061, 0.15986395, 0.20748299, 0.15986395, 0.01785714],
        [0,          0.01785714, 0,          0.05952381, 0.07993197, 0.12159864, 0.18707483, 0.03571429],
        [0,          0.07993197, 0,          0,          0,          0.01785714, 0.03571429, 0.15136054],
    ])

    p = PALD()
    p.pald = pald
    p.threshhold = np.mean(np.diagonal(pald)) / 2

    clusters = p.predict()
    np.testing.assert_equal(clusters, [0, 0, 0, 0, 1, 1, 1, 2])


def test_fit_predict_ex1(setup_data):
    p = PALD()
    clusters = p.fit_predict(setup_data)
    np.testing.assert_equal(clusters, [0, 0, 0, 0, 1, 1, 1, 2])

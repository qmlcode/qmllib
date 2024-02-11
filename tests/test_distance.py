import numpy as np

from qmllib.kernels.distance import l2_distance, manhattan_distance, p_distance


def test_manhattan():

    nfeatures = 5
    n1 = 7
    n2 = 9

    v1 = np.random.random((n1, nfeatures))
    v2 = np.random.random((n2, nfeatures))

    D = manhattan_distance(v1, v2)

    Dtest = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            for k in range(nfeatures):
                Dtest[i, j] += abs(v1[i, k] - v2[j, k])

    assert np.allclose(D, Dtest), "Error in manhattan distance"


def test_l2():

    nfeatures = 5
    n1 = 7
    n2 = 9

    v1 = np.random.random((n1, nfeatures))
    v2 = np.random.random((n2, nfeatures))

    D = l2_distance(v1, v2)

    Dtest = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            for k in range(nfeatures):
                Dtest[i, j] += (v1[i, k] - v2[j, k]) ** 2

    np.sqrt(Dtest, out=Dtest)

    assert np.allclose(D, Dtest), "Error in l2 distance"


def test_p():

    nfeatures = 5
    n1 = 7
    n2 = 9

    v1 = np.random.random((n1, nfeatures))
    v2 = np.random.random((n2, nfeatures))

    D = p_distance(v1, v2, 3)

    Dtest = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            for k in range(nfeatures):
                Dtest[i, j] += abs(v1[i, k] - v2[j, k]) ** 3

    Dtest = Dtest ** (1.0 / 3)

    assert np.allclose(D, Dtest), "Error in p-distance"

    Dfloat = p_distance(v1, v2, 3.0)
    assert np.allclose(D, Dfloat), "Floatingpoint Error in p-distance"


if __name__ == "__main__":
    test_manhattan()
    test_l2()
    test_p()

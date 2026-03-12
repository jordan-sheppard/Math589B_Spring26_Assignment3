import numpy as np

from src.python.modal_lqr import point_coupling


def test_center_misses_even_modes():
    x0 = 0.5
    y0 = 0.5
    assert np.isclose(point_coupling(2, 1, x0, y0), 0.0)
    assert np.isclose(point_coupling(1, 2, x0, y0), 0.0)
    assert not np.isclose(point_coupling(1, 1, x0, y0), 0.0)

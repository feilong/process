import numpy as np
from process.resample import interpolate
import pytest


class TestInterpolation:
    def test_valid(self):
        v = np.arange(24).reshape(2, 3, 4)
        ijk = np.array([
            [-0.5, -0.5, -0.5],
            [ 1.5,  2.5,  3.5],
            [0, 0, 0],
            [1, 2, 3],
        ])
        print(v.shape, ijk.shape)
        interp = interpolate(v, ijk.T)
        assert interp[0] == v[0, 0, 0]
        assert interp[1] == v[1, 2, 3]
        assert interp[2] == v[0, 0, 0]
        assert interp[3] == v[1, 2, 3]
        assert np.all(np.isfinite(interp))

        for idx, itp in zip(ijk, interp):
            assert itp == interpolate(v, idx)

    def test_invalid(self):
        v = np.arange(24).reshape(2, 3, 4)
        ijk = np.array([
            [-0.51, -0.51, -0.51],
            [ 1.51,  2.51,  3.51],
            [-0.51, -0.50, -0.50],
            [-0.50, -0.51, -0.50],
            [-0.50, -0.50, -0.51],
            [ 1.51,  2.50,  3.50],
            [ 1.50,  2.51,  3.50],
            [ 1.50,  2.50,  3.51],
        ])

        with pytest.warns(Warning) as record:
            interp = interpolate(v, ijk.T)
            if not record:
                pytest.fail('Expected a warning!')
        np.testing.assert_array_equal(
            np.isnan(interp),
            np.array([True] * 8),
        )

        for i, (idx, itp) in enumerate(zip(ijk, interp)):
            with pytest.warns(Warning) as record:
                itp2 = interpolate(v, idx)
            assert np.all(np.isnan([itp, itp2])) or itp == itp2

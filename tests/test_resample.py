import numpy as np
import nibabel as nib
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


# sample_bold

from process.resample import sample_bold
from process.fieldmap import fieldmap_shift

SHAPE = (12, 14, 10, 6)


def _linear_data(rng):
    """data[i, j, k, t] = c[t] . (i, j, k) + b[t], which linear interpolation reproduces."""
    c = rng.normal(0, 1, (SHAPE[3], 3))
    b = rng.normal(0, 10, SHAPE[3])
    ijk = np.indices(SHAPE[:3]).reshape(3, -1)
    data = (c @ ijk + b[:, np.newaxis]).T.reshape(SHAPE)
    return data, c, b


def _expected(c, b, coords):
    """coords: (T, 3, D, N) -> (T, N), averaged over D."""
    return (np.einsum('td,tdsn->tsn', c, coords) + b[:, None, None]).mean(axis=1)


def _rigid_vox(rng, n, scale=1.0):
    """Small random rotations and translations, as voxel-to-voxel matrices."""
    mats = []
    for _ in range(n):
        a = rng.normal(0, 0.03 * scale, 3)
        rx = np.array([[1, 0, 0], [0, np.cos(a[0]), -np.sin(a[0])], [0, np.sin(a[0]), np.cos(a[0])]])
        ry = np.array([[np.cos(a[1]), 0, np.sin(a[1])], [0, 1, 0], [-np.sin(a[1]), 0, np.cos(a[1])]])
        rz = np.array([[np.cos(a[2]), -np.sin(a[2]), 0], [np.sin(a[2]), np.cos(a[2]), 0], [0, 0, 1]])
        m = np.eye(4)
        m[:3, :3] = rx @ ry @ rz
        m[:3, 3] = rng.normal(0, 0.5 * scale, 3)
        mats.append(m)
    return np.array(mats)


def _apply(mats, ijk):
    """(T, 4, 4) applied to (3, D, N) -> (T, 3, D, N)."""
    return np.einsum('tab,bsn->tasn', mats[:, :3, :3], ijk) + mats[:, :3, 3, None, None]


def _interior(rng, d, n, margin=2.0):
    lo, hi = margin, np.array(SHAPE[:3]) - 1 - margin
    return rng.uniform(lo, hi[:, None, None], (3, d, n))


def _random_fieldmap(rng, pe_axis, amplitude=1.0):
    """A smooth random field of shifts (in voxels) over the data, in boldref voxels."""
    affine = np.diag([0.25, 0.25, 0.25, 1.0])
    affine[:3, 3] = 1.0  # the data span indices 1 to ~4.5 of the 7 x 7 x 7 grid
    coef = rng.normal(0, amplitude, (7, 7, 7))
    return {'pe_axis': pe_axis, 'levels': [(coef, affine)]}


def test_sample_bold_identity():
    rng = np.random.default_rng(0)
    data, c, b = _linear_data(rng)
    ijk = _interior(rng, 5, 100)
    values, coverage = sample_bold(data, np.tile(np.eye(4), (SHAPE[3], 1, 1)), ijk)
    hmc_coords = np.broadcast_to(ijk, (SHAPE[3],) + ijk.shape)
    np.testing.assert_allclose(values, _expected(c, b, hmc_coords), rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(coverage, 1.0)
    assert values.shape == (SHAPE[3], 100) and values.dtype == np.float64


def test_sample_bold_hmc():
    rng = np.random.default_rng(1)
    data, c, b = _linear_data(rng)
    hmc = _rigid_vox(rng, SHAPE[3])
    ijk = _interior(rng, 3, 200)
    values, coverage = sample_bold(data, hmc, ijk)
    np.testing.assert_allclose(values, _expected(c, b, _apply(hmc, ijk)), rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(coverage, 1.0)


@pytest.mark.parametrize('pe_axis', [0, 1, 2])
def test_sample_bold_sdc(pe_axis):
    """The shift is looked up at the boldref position and added after head motion."""
    rng = np.random.default_rng(2 + pe_axis)
    data, c, b = _linear_data(rng)
    hmc = _rigid_vox(rng, SHAPE[3])
    ijk = _interior(rng, 4, 150)
    fmap = _random_fieldmap(rng, pe_axis)
    shift = fieldmap_shift(fmap, ijk)
    assert 0.1 < np.abs(shift).max() < 2
    coords = _apply(hmc, ijk)
    coords[:, pe_axis] += shift
    values, coverage = sample_bold(data, hmc, ijk, fmap)
    np.testing.assert_allclose(values, _expected(c, b, coords), rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(coverage, 1.0)
    # Not the same as looking the shift up after head motion
    coords_wrong = _apply(hmc, ijk)
    coords_wrong[:, pe_axis] += np.stack([fieldmap_shift(fmap, _) for _ in _apply(hmc, ijk)])
    assert not np.allclose(values, _expected(c, b, coords_wrong), rtol=1e-6, atol=1e-6)


def test_sample_bold_constant_shift():
    """A constant field of +0.5 voxel along j samples 0.5 voxel higher in j."""
    rng = np.random.default_rng(5)
    data, c, b = _linear_data(rng)
    affine = np.diag([0.1, 0.1, 0.1, 1.0])
    affine[:3, 3] = 3  # data well inside a 10 x 10 x 10 grid of equal coefficients
    fmap = {'pe_axis': 1, 'levels': [(np.full((10, 10, 10), 0.5), affine)]}
    eye = np.tile(np.eye(4), (SHAPE[3], 1, 1))
    ijk = _interior(rng, 2, 50)
    values, _ = sample_bold(data, eye, ijk, fmap)
    shifted = ijk.copy()
    shifted[1] += 0.5
    np.testing.assert_allclose(values, sample_bold(data, eye, shifted)[0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(values - sample_bold(data, eye, ijk)[0], 0.5 * c[:, [1]] * np.ones(50),
                               rtol=1e-10, atol=1e-10)


def test_sample_bold_outside_fov():
    """Outside samples take the value at the nearest point within the field of view."""
    rng = np.random.default_rng(6)
    data, c, b = _linear_data(rng)
    n = np.array(SHAPE[:3], dtype=float)
    eye = np.tile(np.eye(4), (SHAPE[3], 1, 1))
    # Location 0: both depths inside. 1: one depth beyond the top face in k.
    # 2: both depths outside (below i, beyond j). 3: on the faces (-0.5, n - 0.5).
    ijk = np.array([
        [[3, 3, -1.2, -0.5], [4, 4, -3.0, 2.0]],
        [[5, 5, 6.0, 4.0], [6, 6, n[1] + 2.0, 4.0]],
        [[4, 4, 5.0, n[2] - 0.5], [5, n[2] + 0.7, 5.0, 3.0]],
    ])
    values, coverage = sample_bold(data, eye, ijk)
    np.testing.assert_allclose(coverage, [1.0, 0.5, 0.0, 1.0])
    clipped = np.clip(ijk, 0, (n - 1)[:, None, None])
    expected = _expected(c, b, np.broadcast_to(clipped, (SHAPE[3],) + ijk.shape))
    np.testing.assert_allclose(values, expected, rtol=1e-10, atol=1e-10)
    # The clipped value differs from extrapolating the linear data
    extrapolated = _expected(c, b, np.broadcast_to(ijk, (SHAPE[3],) + ijk.shape))
    assert not np.allclose(values[:, 1:3], extrapolated[:, 1:3])


def test_sample_bold_coverage_over_time():
    """Coverage counts samples over volumes too: head motion moves a sample out in some volumes."""
    rng = np.random.default_rng(7)
    data, _, _ = _linear_data(rng)
    hmc = np.tile(np.eye(4), (SHAPE[3], 1, 1))
    hmc[:2, 0, 3] = -3.0  # 2 of 6 volumes move points near i = 1 out of the field of view
    ijk = np.array([[[1.0, 6.0]], [[5.0, 5.0]], [[5.0, 5.0]]])
    _, coverage = sample_bold(data, hmc, ijk)
    np.testing.assert_allclose(coverage, [4 / 6, 1.0])


def test_sample_bold_shift_moves_out_of_fov():
    rng = np.random.default_rng(8)
    data, _, _ = _linear_data(rng)
    eye = np.tile(np.eye(4), (SHAPE[3], 1, 1))
    affine = np.diag([0.1, 0.1, 0.1, 1.0])
    affine[:3, 3] = 3
    fmap = {'pe_axis': 1, 'levels': [(np.full((10, 10, 10), 2.0), affine)]}
    ijk = np.array([[[5.0, 5.0]], [[SHAPE[1] - 2.0, 5.0]], [[5.0, 5.0]]])
    _, coverage = sample_bold(data, eye, ijk, fmap)
    np.testing.assert_allclose(coverage, [0.0, 1.0])


def test_sample_bold_proxy(tmp_path):
    """int16 data read through a NIfTI proxy gives the same result as an array."""
    rng = np.random.default_rng(9)
    data = rng.integers(-1000, 1000, SHAPE).astype(np.int16)
    fn = tmp_path / 'bold.nii.gz'
    nib.save(nib.Nifti1Image(data, np.eye(4)), fn)
    hmc = _rigid_vox(rng, SHAPE[3])
    ijk = _interior(rng, 2, 30)
    expected = sample_bold(data.astype(np.float64), hmc, ijk)
    for d in [nib.load(fn).dataobj, np.asanyarray(nib.load(fn).dataobj)]:
        out = sample_bold(d, hmc, ijk)
        for a, e in zip(out, expected):
            np.testing.assert_array_equal(a, e)



@pytest.mark.parametrize('order', [0, 1, 2, 3, 4, 5])
def test_sample_bold_order(order):
    """Coordinates are clipped to the field of view, then interpolated with the given order."""
    from scipy.ndimage import map_coordinates
    rng = np.random.default_rng(10)
    data = rng.normal(0, 1, SHAPE)
    hmc = _rigid_vox(rng, SHAPE[3])
    fmap = _random_fieldmap(rng, 1)
    n = np.array(SHAPE[:3], dtype=float)
    ijk = rng.uniform(-3, n[:, None, None] + 2, (3, 2, 300))  # many points outside
    values, coverage = sample_bold(data, hmc, ijk, fmap, order=order)
    assert np.any(coverage == 1) and np.any((coverage > 0) & (coverage < 1)) and np.any(coverage == 0)

    coords = _apply(hmc, ijk)
    coords[:, 1] += fieldmap_shift(fmap, ijk)
    expected = np.stack([
        map_coordinates(data[..., t], np.clip(coords[t], 0, (n - 1)[:, None, None]),
                        order=order, mode='nearest').mean(axis=0)
        for t in range(SHAPE[3])])
    np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-12)
    if order == 0:  # nearest-neighbor: every sample is a voxel value
        single = sample_bold(data, hmc, ijk[:, :1], fmap, order=0)[0]
        assert np.all(np.isin(single, data))


def test_sample_bold_order_linear_default():
    rng = np.random.default_rng(11)
    data = rng.normal(0, 1, SHAPE)
    hmc = _rigid_vox(rng, SHAPE[3])
    ijk = _interior(rng, 2, 50)
    np.testing.assert_array_equal(sample_bold(data, hmc, ijk)[0], sample_bold(data, hmc, ijk, order=1)[0])
    assert not np.allclose(sample_bold(data, hmc, ijk)[0], sample_bold(data, hmc, ijk, order=3)[0])


def test_sample_bold_cubic_interior():
    """Cubic interpolation reproduces a quadratic exactly, away from the edges."""
    rng = np.random.default_rng(12)
    shape = (40, 40, 40, 2)
    i, j, k = np.indices(shape[:3], dtype=float)
    vol = 0.01 * (i - 20) ** 2 - 0.02 * (j - 15) * (k - 22) + 0.3 * k
    data = np.stack([vol, -vol], axis=-1)
    ijk = rng.uniform(15, 25, (3, 1, 100))
    values, _ = sample_bold(data, np.tile(np.eye(4), (2, 1, 1)), ijk, order=3)
    x, y, z = ijk[:, 0]
    true = 0.01 * (x - 20) ** 2 - 0.02 * (y - 15) * (z - 22) + 0.3 * z
    np.testing.assert_allclose(values, np.stack([true, -true]), atol=1e-6)
    linear, _ = sample_bold(data, np.tile(np.eye(4), (2, 1, 1)), ijk, order=1)
    assert np.abs(linear - np.stack([true, -true])).max() > 1e-3

@pytest.mark.parametrize('kwargs, match', [
    ({'data': np.zeros(SHAPE[:3])}, 'data must be 4D'),
    ({'hmc': np.zeros((SHAPE[3] - 1, 4, 4))}, 'hmc must have shape'),
    ({'hmc': np.zeros((SHAPE[3], 3, 4))}, 'hmc must have shape'),
    ({'ijk': np.zeros((3, 5))}, r'ijk must have shape \(3, D, N\)'),
    ({'ijk': np.zeros((2, 1, 5))}, r'ijk must have shape \(3, D, N\)'),
    ({'order': 6}, 'order must be an integer from 0 to 5'),
    ({'order': -1}, 'order must be an integer from 0 to 5'),
    ({'order': 1.5}, 'order must be an integer from 0 to 5'),
    ({'order': 'linear'}, 'order must be an integer from 0 to 5'),
    ({'order': True}, 'order must be an integer from 0 to 5'),
    ({'order': np.bool_(True)}, 'order must be an integer from 0 to 5'),
    ({'order': 1.0}, 'order must be an integer from 0 to 5'),
    ({'order': np.float64(3)}, 'order must be an integer from 0 to 5'),
    ({'order': None}, 'order must be an integer from 0 to 5'),
    ({'order': np.int64(6)}, 'order must be an integer from 0 to 5'),
    ({'ijk': np.zeros((3, 0, 5))}, r'at least one sample per location \(D > 0\)'),
    ({'data': np.zeros(SHAPE[:3] + (0,)), 'hmc': np.zeros((0, 4, 4))}, 'at least one volume'),
])
def test_sample_bold_bad_input(kwargs, match):
    args = {'data': np.zeros(SHAPE), 'hmc': np.tile(np.eye(4), (SHAPE[3], 1, 1)),
            'ijk': np.ones((3, 1, 5))}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        sample_bold(**args)



@pytest.mark.parametrize('order', [1, np.int64(3), np.int32(0), np.uint8(5)])
def test_sample_bold_integer_orders(order):
    rng = np.random.default_rng(13)
    data = rng.normal(0, 1, SHAPE)
    hmc = np.tile(np.eye(4), (SHAPE[3], 1, 1))
    ijk = _interior(rng, 2, 10)
    np.testing.assert_array_equal(
        sample_bold(data, hmc, ijk, order=order)[0], sample_bold(data, hmc, ijk, order=int(order))[0])


def test_sample_bold_no_locations():
    """N = 0 is allowed and gives empty outputs, without NaNs or errors."""
    data = np.zeros(SHAPE)
    hmc = np.tile(np.eye(4), (SHAPE[3], 1, 1))
    fmap = {'pe_axis': 1, 'levels': [(np.ones((4, 4, 4)), np.eye(4))]}
    for f in [None, fmap]:
        values, coverage = sample_bold(data, hmc, np.zeros((3, 5, 0)), f)
        assert values.shape == (SHAPE[3], 0) and coverage.shape == (0,)

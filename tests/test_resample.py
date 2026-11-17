import os

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


# resample_run

import json

import nitransforms as nt

from process.fieldmap import load_fieldmap
from process.resample import resample_run

# LAS, 3 x 3 x 3.5 mm, slightly oblique
RUN_AFFINE = np.array([
    [-3.0, 0.1, 0.0, 20.0],
    [0.0, 3.0, 0.2, -25.0],
    [0.0, -0.1, 3.5, -15.0],
    [0.0, 0.0, 0.0, 1.0]])


def _itk_matrix(fn):
    """The matrices as stored in an ITK text file (which has limited precision)."""
    return nt.linear.LinearTransformsMapping.from_filename(fn, fmt='itk').matrix


@pytest.fixture
def run_files(tmp_path):
    """A raw BOLD run with linear intensities, and fMRIPrep-like transforms and fieldmap."""
    rng = np.random.default_rng(0)
    data, c, b = _linear_data(rng)
    func = tmp_path / 'bids' / 'sub-01' / 'func'
    func.mkdir(parents=True)
    bold_fn = func / 'sub-01_task-x_bold.nii.gz'
    nib.save(nib.Nifti1Image(data, RUN_AFFINE), bold_fn)
    (func / 'sub-01_task-x_bold.json').write_text(json.dumps(
        {'PhaseEncodingDirection': 'j-', 'TotalReadoutTime': 0.05}))

    hmc_world = RUN_AFFINE @ _rigid_vox(rng, SHAPE[3]) @ np.linalg.inv(RUN_AFFINE)
    hmc_fn = tmp_path / 'sub-01_task-x_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt'
    nt.linear.LinearTransformsMapping(hmc_world).to_filename(hmc_fn, fmt='itk')
    coreg = _rigid_vox(rng, 1, scale=3.0)[0]
    coreg_fn = tmp_path / 'sub-01_task-x_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt'
    nt.linear.Affine(coreg).to_filename(coreg_fn, fmt='itk')

    center = RUN_AFFINE @ np.append((np.array(SHAPE[:3]) - 1) / 2, 1)
    coef_affine = np.diag([20.0, 20.0, 20.0, 1.0])
    coef_affine[:3, 3] = center[:3] - 50.0  # a 6 x 6 x 6 grid centered on the data
    coeff_fn = tmp_path / 'sub-01_fmapid-auto00000_desc-coeff_fieldmap.nii.gz'
    nib.save(nib.Nifti1Image(rng.normal(0, 10, (6, 6, 6)), coef_affine), coeff_fn)
    xfm_fn = tmp_path / 'sub-01_task-x_from-boldref_to-auto00000_mode-image_xfm.txt'
    nt.linear.Affine(_rigid_vox(rng, 1)[0]).to_filename(xfm_fn, fmt='itk')

    # The fieldmap's grid: the BOLD grid with 2 more voxels on every side
    ref_affine = RUN_AFFINE.copy()
    ref_affine[:3, 3] = (RUN_AFFINE @ [-2, -2, -2, 1])[:3]
    ref_fn = tmp_path / 'sub-01_fmapid-auto00000_desc-preproc_fieldmap.nii.gz'
    ref_shape = tuple(np.array(SHAPE[:3]) + 4)
    nib.save(nib.Nifti1Image(np.zeros(ref_shape, dtype=np.float32), ref_affine), ref_fn)

    run = {
        'label': 'task-x', 'bold_fn': str(bold_fn), 'hmc_fn': str(hmc_fn),
        'coreg_fn': str(coreg_fn),
        'fieldmap': {'xfm_fn': str(xfm_fn), 'coeff_fns': [str(coeff_fn)], 'ref_fn': str(ref_fn)}}
    return run, str(tmp_path / 'bids'), c, b


def _t1w_coords(run, ijk):
    """Boldref voxel positions (3, D, N) -> T1w world coordinates (D, N, 3)."""
    to_t1w = np.linalg.inv(_itk_matrix(run['coreg_fn'])[0]) @ RUN_AFFINE
    return np.moveaxis(_apply(to_t1w[None], ijk)[0], 0, -1)


@pytest.mark.filterwarnings('error')
@pytest.mark.parametrize('sdc', [True, False])
def test_resample_run(run_files, sdc):
    run, bids_dir, c, b = run_files
    rng = np.random.default_rng(1)
    ijk = _interior(rng, 3, 50, margin=3.0)
    hmc = np.linalg.inv(RUN_AFFINE) @ _itk_matrix(run['hmc_fn']) @ RUN_AFFINE
    moved = _apply(hmc, ijk)
    if sdc:
        fieldmap = load_fieldmap(
            run['fieldmap']['coeff_fns'], run['fieldmap']['xfm_fn'], RUN_AFFINE, 'j-', 0.05)
        shift = fieldmap_shift(fieldmap, ijk)
        assert np.abs(shift).max() > 0.2
        moved[:, 1] += shift
    else:
        run = dict(run, fieldmap=None)
        bids_dir = 'not-read-without-a-fieldmap'

    values, coverage = resample_run(run, bids_dir, _t1w_coords(run, ijk))
    assert values.shape == (SHAPE[3], 50) and values.dtype == np.float64
    np.testing.assert_allclose(values, _expected(c, b, moved), atol=1e-8)
    np.testing.assert_array_equal(coverage, 1.0)


def test_resample_run_order(run_files):
    run, bids_dir, c, b = run_files
    coords = _t1w_coords(run, _interior(np.random.default_rng(2), 2, 10, margin=3.0))
    linear, _ = resample_run(run, bids_dir, coords)
    nearest, _ = resample_run(run, bids_dir, coords, order=0)
    assert not np.allclose(linear, nearest)


@pytest.mark.parametrize('shape', [(5, 3), (3, 2, 5), (2, 5, 4), (3,)])
def test_resample_run_bad_coords(run_files, shape):
    run, bids_dir, c, b = run_files
    with pytest.raises(ValueError, match='coords'):
        resample_run(run, bids_dir, np.zeros(shape))


def test_resample_run_outside_fieldmap(run_files, tmp_path):
    """Warn when many positions are in slices that the fieldmap does not cover."""
    run, bids_dir, c, b = run_files
    ref = nib.load(run['fieldmap']['ref_fn'])
    small_fn = str(tmp_path / 'small_desc-preproc_fieldmap.nii.gz')
    nib.save(ref.slicer[:, :, :8], small_fn)  # covers BOLD slices up to k = 5.5, roughly
    small = dict(run, fieldmap=dict(run['fieldmap'], ref_fn=small_fn))
    coords = _t1w_coords(run, _interior(np.random.default_rng(3), 2, 200, margin=1.0))  # k in [1, 8]

    expected, _ = resample_run(run, bids_dir, coords)
    with pytest.warns(UserWarning, match=r"task-x: \d+\.\d% of the sample positions are outside"):
        values, _ = resample_run(small, bids_dir, coords)
    np.testing.assert_array_equal(values, expected)  # the correction itself is unchanged

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        resample_run(small, bids_dir, coords, max_outside_fieldmap=1.0)
        resample_run(dict(run, fieldmap=dict(run['fieldmap'], ref_fn=None)), bids_dir, coords)


# resample_surfaces

from scipy import sparse

import process.resample
from process.resample import resample_surfaces

N_NATIVE = {'l': 30, 'r': 40}
N_TEMPLATE = 12
COMBINATIONS = [('onavg-ico32', '1step_pial_overlap'), ('native', '1step_pial_overlap')]


@pytest.fixture
def surface_inputs(run_files, tmp_path, monkeypatch):
    """Surfaces and transforms for the run of `run_files`; `find_runs` returns that run."""
    run, bids_dir, c, b = run_files
    rng = np.random.default_rng(5)
    surfaces, xforms = {}, {}
    for lr in 'lr':
        ijk = _interior(rng, 2, N_NATIVE[lr], margin=3.0)
        ijk[:, 1] = ijk[:, 0] + rng.normal(0, 0.5, (3, N_NATIVE[lr]))
        surfaces[lr] = tuple(_t1w_coords(run, ijk))  # white, pial
        xform = sparse.random(N_NATIVE[lr], N_TEMPLATE, density=0.3, random_state=rng, format='lil')
        xform[:, 0] = 0  # a template vertex without native vertices
        xform[0, 1:] = 1.0
        xforms[lr] = xform.tocsr()
        fn = tmp_path / 'xforms' / 'onavg-ico32' / f'01_overlap_{lr}h.npz'
        fn.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(fn, xforms[lr])
    calls = []

    def find_runs(fmriprep_dir, bids_dir, sid, filter_=None, **kwargs):
        calls.append((fmriprep_dir, sid, filter_, kwargs))
        return [run]

    monkeypatch.setattr(process.resample, 'find_runs', find_runs)
    monkeypatch.setattr(process.resample, 'load_surfaces',
                        lambda fmriprep_dir, sid, lr, fs_dir=None: surfaces[lr])
    dirs = dict(out_dir=str(tmp_path / 'resampled'), xform_dir=str(tmp_path / 'xforms'),
                coverage_dir=str(tmp_path / 'confounds' / 'coverage'))
    return run, bids_dir, surfaces, xforms, dirs, calls


def test_resample_surfaces(surface_inputs):
    run, bids_dir, surfaces, xforms, dirs, calls = surface_inputs
    resample_surfaces('01', bids_dir, 'fmriprep', combinations=COMBINATIONS,
                      no_fieldmap=('task-y',), **dirs)
    assert calls == [('fmriprep', '01', None, {'no_fieldmap': ('task-y',), 'fieldmap_prefix': None})]

    ratios = np.array([0.1, 0.3, 0.5, 0.7, 0.9])[:, None, None]
    for lr in 'lr':
        white, pial = surfaces[lr]
        native, native_cov = resample_run(run, bids_dir, white * ratios + pial * (1 - ratios))
        w = xforms[lr].toarray()
        w[:, 1:] /= w[:, 1:].sum(axis=0)
        for space, expected, expected_cov in [
                ('native', native, native_cov), ('onavg-ico32', native @ w, native_cov @ w)]:
            rel = f'{space}/{lr}-cerebrum/1step_pial_overlap/sub-01_task-x.npy'
            data = np.load(f"{dirs['out_dir']}/{rel}")
            coverage = np.load(f"{dirs['coverage_dir']}/{rel}")
            assert data.dtype == coverage.dtype == np.float64
            assert data.shape == (SHAPE[3], expected.shape[1]) and coverage.shape == data.shape[1:]
            np.testing.assert_allclose(data, expected, atol=1e-9)
            np.testing.assert_allclose(coverage, expected_cov, atol=1e-12)
        template = np.load(f"{dirs['out_dir']}/onavg-ico32/{lr}-cerebrum/1step_pial_overlap/sub-01_task-x.npy")
        np.testing.assert_array_equal(template[:, 0], 0.0)  # no native vertices


def test_resample_surfaces_skips_existing(surface_inputs, monkeypatch):
    run, bids_dir, surfaces, xforms, dirs, calls = surface_inputs
    resample_surfaces('01', bids_dir, 'fmriprep', combinations=COMBINATIONS, **dirs)
    fn = f"{dirs['out_dir']}/onavg-ico32/r-cerebrum/1step_pial_overlap/sub-01_task-x.npy"
    other = f"{dirs['out_dir']}/onavg-ico32/l-cerebrum/1step_pial_overlap/sub-01_task-x.npy"
    expected = np.load(fn)
    os.remove(fn)
    np.save(other, np.zeros(3))  # existing outputs are not rewritten
    resample_surfaces('01', bids_dir, 'fmriprep', combinations=COMBINATIONS, **dirs)
    np.testing.assert_array_equal(np.load(fn), expected)
    np.testing.assert_array_equal(np.load(other), np.zeros(3))

    # Only the missing file of a pair is written
    coverage_fn = fn.replace(dirs['out_dir'], dirs['coverage_dir'])
    expected_coverage = np.load(coverage_fn)
    os.remove(coverage_fn)
    np.save(fn, np.ones(3))
    resample_surfaces('01', bids_dir, 'fmriprep', combinations=COMBINATIONS, **dirs)
    np.testing.assert_array_equal(np.load(coverage_fn), expected_coverage)
    np.testing.assert_array_equal(np.load(fn), np.ones(3))
    os.remove(fn)
    np.save(coverage_fn, np.ones(3))
    resample_surfaces('01', bids_dir, 'fmriprep', combinations=COMBINATIONS, **dirs)
    np.testing.assert_array_equal(np.load(fn), expected)
    np.testing.assert_array_equal(np.load(coverage_fn), np.ones(3))

    # Nothing is loaded when everything exists
    monkeypatch.setattr(process.resample, 'load_surfaces', None)
    resample_surfaces('01', bids_dir, 'fmriprep', combinations=COMBINATIONS, **dirs)


def test_resample_surfaces_errors(surface_inputs):
    run, bids_dir, surfaces, xforms, dirs, calls = surface_inputs
    with pytest.raises(ValueError, match='Unsupported tag'):
        resample_surfaces('01', bids_dir, 'fmriprep', combinations=[('onavg-ico32', '1step_pial_area')], **dirs)
    for n_depths in [1.1, 5.0, 0, -1, True, '5', None, np.float64(5)]:
        with pytest.raises(ValueError, match='n_depths'):
            resample_surfaces('01', bids_dir, 'fmriprep', combinations=COMBINATIONS, n_depths=n_depths, **dirs)
    assert not calls
    assert not os.path.exists(dirs['out_dir'])
    with pytest.raises(FileNotFoundError, match='xform step'):
        resample_surfaces('01', bids_dir, 'fmriprep', combinations=[('onavg-ico64', '1step_pial_overlap')], **dirs)


def test_resample_surfaces_n_depths(surface_inputs):
    run, bids_dir, surfaces, xforms, dirs, calls = surface_inputs
    resample_surfaces('01', bids_dir, 'fmriprep', combinations=[('native', '1step_pial_overlap')],
                      n_depths=np.int64(1), **dirs)
    white, pial = surfaces['l']
    expected, _ = resample_run(run, bids_dir, ((white + pial) / 2)[None])
    data = np.load(f"{dirs['out_dir']}/native/l-cerebrum/1step_pial_overlap/sub-01_task-x.npy")
    np.testing.assert_allclose(data, expected, atol=1e-9)


# check_sdc

from process.resample import check_sdc, check_outputs


def _truth(ijk):
    """A smooth image in boldref voxel coordinates: three blobs on a ramp."""
    out = 100.0 + 2.0 * ijk[0]
    for center, width, height in [((4, 5, 4), 2.5, 400.0), ((8, 9, 6), 2.0, 300.0), ((6, 3, 3), 1.5, 500.0)]:
        d2 = sum((ijk[i] - center[i]) ** 2 for i in range(3))
        out = out + height * np.exp(-d2 / (2 * width ** 2))
    return out


@pytest.fixture
def distorted_run(run_files):
    """The run of `run_files`, with raw data that are a moved and distorted `_truth`."""
    run, bids_dir, c, b = run_files
    hmc = np.linalg.inv(RUN_AFFINE) @ _itk_matrix(run['hmc_fn']) @ RUN_AFFINE
    fieldmap = load_fieldmap(
        run['fieldmap']['coeff_fns'], run['fieldmap']['xfm_fn'], RUN_AFFINE, 'j-', 0.05)
    q = np.indices(SHAPE[:3]).reshape(3, -1).astype(np.float64)
    data = np.empty(SHAPE)
    for t in range(SHAPE[3]):
        # Solve q = hmc[t] p + shift(p) e_j for the boldref position p of each raw voxel
        inv = np.linalg.inv(hmc[t])
        p = inv[:3, :3] @ q + inv[:3, 3:]
        for _ in range(30):
            moved = q.copy()
            moved[1] -= fieldmap_shift(fieldmap, p)
            p = inv[:3, :3] @ moved + inv[:3, 3:]
        data[..., t] = _truth(p).reshape(SHAPE[:3])
    nib.save(nib.Nifti1Image(data, RUN_AFFINE), run['bold_fn'])
    ref_fn = os.path.join(os.path.dirname(run['hmc_fn']), 'sub-01_task-x_desc-coreg_boldref.nii.gz')
    truth = _truth(np.indices(SHAPE[:3]).astype(np.float64))
    nib.save(nib.Nifti1Image(truth, RUN_AFFINE), ref_fn)
    return run, bids_dir, ref_fn


def test_check_sdc(distorted_run):
    run, bids_dir, ref_fn = distorted_run
    result = check_sdc(run, bids_dir)
    assert set(result) == {'r_sdc', 'r_nosdc', 'shift_p1', 'shift_p99'}
    assert all(isinstance(_, float) for _ in result.values())
    assert result['r_sdc'] > 0.998  # limited by linear interpolation of the blobs
    assert result['r_nosdc'] < result['r_sdc'] - 0.002
    assert result['shift_p99'] - result['shift_p1'] > 0.2
    # A subset of the volumes gives nearly the same answer
    assert check_sdc(run, bids_dir, n_frames=2)['r_sdc'] == pytest.approx(result['r_sdc'], abs=1e-3)


def test_check_sdc_wrong_sign(distorted_run):
    """A phase-encoding direction with the wrong sign is worse than no correction."""
    run, bids_dir, ref_fn = distorted_run
    json_fn = run['bold_fn'].replace('.nii.gz', '.json')
    with open(json_fn, 'w') as f:
        json.dump({'PhaseEncodingDirection': 'j', 'TotalReadoutTime': 0.05}, f)
    result = check_sdc(run, bids_dir)
    assert result['r_sdc'] < result['r_nosdc']


def test_check_sdc_errors(distorted_run):
    run, bids_dir, ref_fn = distorted_run
    with pytest.raises(ValueError, match='no fieldmap'):
        check_sdc(dict(run, fieldmap=None), bids_dir)
    os.remove(ref_fn)
    with pytest.raises(FileNotFoundError, match='coreg_boldref'):
        check_sdc(run, bids_dir)


def test_check_sdc_part_entity(distorted_run):
    """The reference keeps `part`, which the transforms drop."""
    run, bids_dir, ref_fn = distorted_run
    expected = check_sdc(run, bids_dir)
    part_fn = run['bold_fn'].replace('_bold.nii.gz', '_part-mag_bold.nii.gz')
    os.rename(run['bold_fn'], part_fn)
    os.rename(run['bold_fn'].replace('.nii.gz', '.json'), part_fn.replace('.nii.gz', '.json'))
    run = dict(run, bold_fn=part_fn, label='task-x_part-mag')
    with pytest.raises(FileNotFoundError, match='part-mag_desc-coreg_boldref'):
        check_sdc(run, bids_dir)
    os.rename(ref_fn, ref_fn.replace('_desc-coreg', '_part-mag_desc-coreg'))
    assert check_sdc(run, bids_dir) == expected


@pytest.fixture
def checked_outputs(surface_inputs):
    run, bids_dir, surfaces, xforms, dirs, calls = surface_inputs
    resample_surfaces('01', bids_dir, 'fmriprep', combinations=COMBINATIONS, **dirs)
    calls.clear()

    def check(**kwargs):
        return check_outputs('01', bids_dir, 'fmriprep', combinations=COMBINATIONS, **dirs, **kwargs)
    return check, dirs, calls


REL = 'onavg-ico32/l-cerebrum/1step_pial_overlap/sub-01_task-x.npy'


def test_check_outputs(checked_outputs):
    check, dirs, calls = checked_outputs
    report = check(filter_=len, no_fieldmap=('task-y',), fieldmap_prefix={'a': 'b'})
    assert calls == [('fmriprep', '01', len, {'no_fieldmap': ('task-y',), 'fieldmap_prefix': {'a': 'b'}})]
    json.dumps(report, allow_nan=False)
    assert report['errors'] == []
    assert list(report['runs']) == ['task-x'] and report['runs']['task-x']['n_frames'] == SHAPE[3]
    outputs = report['runs']['task-x']['outputs']
    assert len(outputs) == 4
    assert outputs[REL]['n_vertices'] == N_TEMPLATE
    assert outputs['native/r-cerebrum/1step_pial_overlap/sub-01_task-x.npy']['n_vertices'] == N_NATIVE['r']
    # The template vertex without native vertices has no signal and no coverage
    assert outputs[REL]['n_low_signal'] >= 1 and outputs[REL]['n_low_coverage'] >= 1
    assert f"{REL}: {outputs[REL]['n_low_coverage']} vertices with low coverage" in report['warnings']
    # Runs selected by the caller are not selected again
    calls.clear()
    assert check(runs=[], filter_=len) == {'runs': {}, 'errors': [], 'warnings': []} and calls == []
    # Thresholds
    mean = np.load(os.path.join(dirs['out_dir'], REL)).mean(axis=0)
    coverage = np.load(os.path.join(dirs['coverage_dir'], REL))
    for low_signal, low_coverage in [(0.05, 0.5), (0.5, 0.9), (2.0, -1.0)]:
        out = check(low_signal=low_signal, low_coverage=low_coverage)['runs']['task-x']['outputs'][REL]
        assert out['median_mean'] == np.median(mean)
        assert out['n_low_signal'] == np.sum(mean < low_signal * np.median(mean))
        assert out['n_low_coverage'] == np.sum(coverage < low_coverage)
    assert out['n_low_coverage'] == 0


def _rewrite(fn, func):
    data = func(np.load(fn))
    np.save(fn, data)


@pytest.mark.parametrize('name, which, func, message', [
    ('nan', 'out_dir', lambda d: np.where(np.arange(d.size).reshape(d.shape) == 3, np.nan, d),
     f'{REL}: 1 nonfinite values'),
    ('inf', 'out_dir', lambda d: np.where(np.arange(d.size).reshape(d.shape) < 2, -np.inf, d),
     f'{REL}: 2 nonfinite values'),
    ('dtype', 'out_dir', lambda d: d.astype(np.float32), f'{REL}: dtype float32, expected float64'),
    ('frames', 'out_dir', lambda d: d[:-1],
     f'{REL}: shape {(SHAPE[3] - 1, N_TEMPLATE)}, expected {(SHAPE[3], N_TEMPLATE)}'),
    ('vertices', 'out_dir', lambda d: d[:, :-1],
     f'{REL}: shape {(SHAPE[3], N_TEMPLATE - 1)}, expected {(SHAPE[3], N_TEMPLATE)}'),
    ('cov-inf', 'coverage_dir', lambda d: np.where(np.arange(d.size) == 1, np.inf, d),
     f'{REL}: coverage: 1 nonfinite values'),
    ('cov-shape', 'coverage_dir', lambda d: d[None],
     f'{REL}: coverage: shape {(1, N_TEMPLATE)}, expected {(N_TEMPLATE,)}'),
    ('cov-high', 'coverage_dir', lambda d: np.where(np.arange(d.size) == 1, 1.001, d),
     f'{REL}: coverage: values outside [0, 1]'),
    ('cov-low', 'coverage_dir', lambda d: np.where(np.arange(d.size) == 1, -0.001, d),
     f'{REL}: coverage: values outside [0, 1]'),
])
def test_check_outputs_errors(checked_outputs, name, which, func, message):
    check, dirs, calls = checked_outputs
    _rewrite(os.path.join(dirs[which], REL), func)
    report = check()
    assert len(report['errors']) == 1 and report['errors'][0].startswith(message)
    # The pair is left out of the summaries; the other files are still checked
    outputs = report['runs']['task-x']['outputs']
    assert REL not in outputs and len(outputs) == 3
    json.dumps(report, allow_nan=False)


def test_check_outputs_coverage_tolerance(checked_outputs):
    check, dirs, calls = checked_outputs
    _rewrite(os.path.join(dirs['coverage_dir'], REL),
             lambda d: np.where(np.arange(d.size) == 1, 1 + 1e-9, d))
    assert check()['errors'] == []


def test_check_outputs_missing_and_unreadable(checked_outputs):
    check, dirs, calls = checked_outputs
    os.remove(os.path.join(dirs['out_dir'], REL))
    other = REL.replace('l-cerebrum', 'r-cerebrum')
    with open(os.path.join(dirs['coverage_dir'], other), 'wb') as f:
        f.write(b'not an array')
    errors = check()['errors']
    assert len(errors) == 2
    assert errors[0] == f'{REL}: missing'
    assert errors[1].startswith(f'{other}: coverage: unreadable')


def test_check_outputs_unreadable_inputs(checked_outputs, surface_inputs):
    check, dirs, calls = checked_outputs
    run = surface_inputs[0]
    os.remove(os.path.join(dirs['xform_dir'], 'onavg-ico32', '01_overlap_lh.npz'))
    with open(run['bold_fn'], 'wb') as f:
        f.write(b'not an image')
    report = check()
    assert len(report['errors']) == 2
    assert report['errors'][0].startswith('onavg-ico32 lh: cannot get the number of vertices')
    assert report['errors'][1].startswith('task-x: cannot get the number of volumes')
    # Files are still checked, without the unknown dimensions
    assert report['runs']['task-x']['n_frames'] is None
    assert len(report['runs']['task-x']['outputs']) == 4
    _rewrite(os.path.join(dirs['out_dir'], REL), lambda d: d[0])
    errors = check()['errors']
    assert len(errors) == 3 and errors[2].startswith(f"{REL}: shape ({N_TEMPLATE},), expected ('any', 'any')")


def test_check_outputs_malformed_files(checked_outputs, monkeypatch):
    from xml.parsers.expat import ExpatError
    check, dirs, calls = checked_outputs

    def malformed(*args, **kwargs):
        raise ExpatError('no element found')
    monkeypatch.setattr('process.resample.load_surfaces', malformed)
    # A truncated overlap archive, and an NPZ archive in place of an array
    xform_fn = os.path.join(dirs['xform_dir'], 'onavg-ico32', '01_overlap_lh.npz')
    with open(xform_fn, 'rb') as f:
        head = f.read(40)
    with open(xform_fn, 'wb') as f:
        f.write(head)
    with open(os.path.join(dirs['out_dir'], REL), 'wb') as f:
        np.savez(f, a=np.zeros(3))
    report = check()
    json.dumps(report, allow_nan=False)
    errors = report['errors']
    assert len(errors) == 4
    assert errors[0].startswith('native lh: cannot get the number of vertices (ExpatError')
    assert errors[1].startswith('onavg-ico32 lh: cannot get the number of vertices (BadZipFile')
    assert errors[2].startswith('native rh: cannot get the number of vertices (ExpatError')
    assert errors[3] == f'{REL}: not an NPY array'
    assert len(report['runs']['task-x']['outputs']) == 3

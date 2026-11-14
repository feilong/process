import nibabel as nib
import nitransforms as nt
import numpy as np
import pytest

from process.fieldmap import load_fieldmap, fieldmap_shift

PE_DIRECTIONS = ['i', 'i-', 'j', 'j-', 'k', 'k-']

BOLD_SHAPE = (20, 22, 14)
# LAS, 3 x 3 x 3.5 mm, slightly oblique
BOLD_AFFINE = np.array([
    [-3.0, 0.1, 0.0, 30.0],
    [0.0, 3.0, 0.2, -40.0],
    [0.0, -0.1, 3.5, -20.0],
    [0.0, 0.0, 0.0, 1.0]])
# Rigid transform; its inverse maps BOLD reference to fieldmap world coordinates
XFM = np.array([
    [0.995, -0.0998, 0.0, 3.0],
    [0.0998, 0.995, 0.0, -2.0],
    [0.0, 0.0, 1.0, 5.0],
    [0.0, 0.0, 0.0, 1.0]])


@pytest.fixture
def fmap_files(tmp_path):
    """Two levels of random B-spline coefficients (in Hz), and the transform."""
    rng = np.random.default_rng(0)
    coeff_fns = []
    for level, (shape, spacing) in enumerate([((5, 6, 5), 40.0), ((9, 10, 8), 20.0)]):
        affine = np.diag([spacing, spacing, spacing, 1.0])
        affine[:3, 3] = -spacing * (np.array(shape) - 1) / 2
        fn = tmp_path / f'sub-01_fmapid-auto00000_desc-coeff{level}_fieldmap.nii.gz'
        nib.save(nib.Nifti1Image(rng.normal(0, 20, shape), affine), fn)
        coeff_fns.append(str(fn))
    xfm_fn = tmp_path / 'sub-01_task-x_from-boldref_to-auto00000_mode-image_xfm.txt'
    nt.linear.Affine(XFM).to_filename(xfm_fn, fmt='itk')
    return coeff_fns, str(xfm_fn)


def test_load_fieldmap(fmap_files):
    coeff_fns, xfm_fn = fmap_files
    fmap = load_fieldmap(coeff_fns, xfm_fn, BOLD_AFFINE, 'j-', 0.05)
    assert fmap['pe_axis'] == 1
    assert len(fmap['levels']) == 2
    for (coef, affine), fn in zip(fmap['levels'], coeff_fns):
        img = nib.load(fn)
        np.testing.assert_allclose(coef, img.get_fdata() * -0.05)
        # BOLD voxel -> BOLD world -> fieldmap world -> coefficient grid
        expected = np.linalg.inv(img.affine) @ np.linalg.inv(XFM) @ BOLD_AFFINE
        np.testing.assert_allclose(affine, expected)



# Tests that do not need sdcflows: compare with the analytic cubic B-spline.

def _cubic_bspline(x):
    """Centered cubic B-spline (Unser 1999, Eq. 6)."""
    x = np.abs(x)
    return np.where(x < 1, 2 / 3 - x**2 + x**3 / 2, np.where(x < 2, (2 - x)**3 / 6, 0.0))


def _reference_shift(levels, ijk):
    """sum_k c_k B(g0 - k0) B(g1 - k1) B(g2 - k2), summed over levels, by brute force."""
    points = np.asarray(ijk, dtype=np.float64).reshape(3, -1)
    total = np.zeros(points.shape[1])
    for coef, affine in levels:
        grid = affine[:3, :3] @ points + affine[:3, 3:]
        w = [_cubic_bspline(grid[d][:, np.newaxis] - np.arange(n)) for d, n in enumerate(coef.shape)]
        total += np.einsum('ni,nj,nk,ijk->n', *w, coef)
    return total.reshape(np.shape(ijk)[1:])


def _random_level(rng, shape, affine):
    return rng.normal(0, 1, shape), np.asarray(affine, dtype=np.float64)


def test_fieldmap_shift_analytic_single_level():
    """Interior, boundary, within support beyond the array, and beyond support."""
    rng = np.random.default_rng(1)
    shape = (5, 6, 4)
    fmap = {'pe_axis': 1, 'levels': [_random_level(rng, shape, np.eye(4))]}
    n = np.array(shape, dtype=float)[:, np.newaxis]

    interior = rng.uniform(0, 1, (3, 200)) * (n - 1)
    edges = np.array([0.0, -0.5, -1.0, -1.5, -1.99])[np.newaxis]
    boundary = np.concatenate([np.repeat(interior[:, :5], 5, axis=1)] * 6, axis=1)
    for axis in range(3):  # each axis, below the first and above the last index
        cols = slice(axis * 50, axis * 50 + 25)
        boundary[axis, cols] = np.repeat(edges, 5, axis=0).ravel()
        cols = slice(axis * 50 + 25, axis * 50 + 50)
        boundary[axis, cols] = (n[axis] - 1 - np.repeat(edges, 5, axis=0)).ravel()
    for points in [interior, boundary]:
        np.testing.assert_allclose(
            fieldmap_shift(fmap, points), _reference_shift(fmap['levels'], points),
            rtol=1e-12, atol=1e-12)
    # Within support but beyond the array, the field is not zero
    assert np.all(fieldmap_shift(fmap, boundary) != 0)

    # More than 2 grid spacings beyond the outermost coefficients: exactly zero
    outside = np.repeat(interior[:, :4], 6, axis=1)
    for axis in range(3):
        outside[axis, axis * 8:axis * 8 + 4] = [-2.001, -2.5, -10.0, -1e6]
        outside[axis, axis * 8 + 4:axis * 8 + 8] = n[axis] - 1 + np.array([2.001, 2.5, 10.0, 1e6])
    assert np.all(fieldmap_shift(fmap, outside) == 0)
    # At exactly 2 spacings the spline is 0, up to rounding
    at_edge = outside.copy()
    at_edge[at_edge < -1.5] = -2.0
    for axis in range(3):
        at_edge[axis, at_edge[axis] > n[axis] + 0.5] = n[axis] - 1 + 2.0
    np.testing.assert_allclose(fieldmap_shift(fmap, at_edge), 0, atol=1e-15)


def test_fieldmap_shift_analytic_levels():
    """Several levels with general (oblique, anisotropic, flipped) affines."""
    rng = np.random.default_rng(2)
    affine1 = np.array([
        [0.08, 0.01, 0.0, 1.0],
        [-0.01, 0.07, 0.02, 0.5],
        [0.0, -0.02, -0.09, 3.5],
        [0.0, 0.0, 0.0, 1.0]])
    levels = [
        _random_level(rng, (4, 5, 4), affine1),
        _random_level(rng, (8, 9, 7), affine1 @ np.diag([2.0, 2.0, 2.0, 1.0])),
        _random_level(rng, (3, 3, 3), np.diag([0.05, 0.05, 0.05, 1.0])),
    ]
    fmap = {'pe_axis': 0, 'levels': levels}
    ijk = rng.uniform(-30, 70, (3, 500))
    expected = _reference_shift(levels, ijk)
    assert np.count_nonzero(expected) > 300
    np.testing.assert_allclose(fieldmap_shift(fmap, ijk), expected, rtol=1e-12, atol=1e-12)
    # The levels add up
    total = sum(fieldmap_shift({'pe_axis': 0, 'levels': [_]}, ijk) for _ in levels)
    np.testing.assert_allclose(fieldmap_shift(fmap, ijk), total, rtol=1e-12, atol=1e-12)


def test_fieldmap_shift_shapes():
    rng = np.random.default_rng(3)
    fmap = {'pe_axis': 2, 'levels': [_random_level(rng, (5, 5, 5), np.diag([0.5, 0.5, 0.5, 1.0]))]}
    ijk = rng.uniform(-2, 10, (3, 4, 5, 6))
    expected = _reference_shift(fmap['levels'], ijk)
    for x, y in [
        (ijk, expected),                                    # (3, A, B, C)
        (ijk[..., 0], expected[..., 0]),                    # (3, A, B)
        (ijk.reshape(3, -1), expected.ravel()),             # (3, N)
        (ijk[:, 1, 2, 3], expected[1, 2, 3]),               # (3,)
        (ijk[:, 1, 2, 3].tolist(), expected[1, 2, 3]),      # list
        (ijk.reshape(3, -1)[:, :0], np.zeros(0)),           # no points
    ]:
        out = fieldmap_shift(fmap, x)
        assert out.shape == np.shape(y)
        np.testing.assert_allclose(out, y, rtol=1e-12, atol=1e-12)
    # Non-contiguous input gives the same values
    np.testing.assert_allclose(
        fieldmap_shift(fmap, ijk.transpose(0, 3, 2, 1)), expected.transpose(2, 1, 0), rtol=1e-12)


@pytest.mark.parametrize('ijk', [np.zeros((2, 5)), np.zeros((4,)), np.float64(1.0), np.zeros((5, 3))])
def test_fieldmap_shift_bad_shape(ijk):
    fmap = {'pe_axis': 0, 'levels': [(np.zeros((4, 4, 4)), np.eye(4))]}
    with pytest.raises(ValueError, match='ijk must have shape'):
        fieldmap_shift(fmap, ijk)


def test_load_fieldmap_then_shift(fmap_files):
    """Shifts from loaded files: analytic B-spline in Hz, times the signed readout time."""
    coeff_fns, xfm_fn = fmap_files
    ijk = np.indices(BOLD_SHAPE)[:, ::3, ::3, ::2]
    hz = None
    for pe_direction in PE_DIRECTIONS:
        fmap = load_fieldmap(coeff_fns, xfm_fn, BOLD_AFFINE, pe_direction, 0.05)
        assert fmap['pe_axis'] == 'ijk'.index(pe_direction[0])
        if hz is None:
            hz = _reference_shift([(nib.load(fn).get_fdata(), affine)
                                   for fn, (_, affine) in zip(coeff_fns, fmap['levels'])], ijk)
            assert np.abs(hz).max() > 1
        sign = -1 if pe_direction.endswith('-') else 1
        np.testing.assert_allclose(fieldmap_shift(fmap, ijk), sign * 0.05 * hz, rtol=1e-12, atol=1e-12)


# Invalid inputs to the loader

@pytest.mark.parametrize('coeff_fns', [[], (), iter([])])
def test_load_fieldmap_no_coefficients(fmap_files, coeff_fns):
    _, xfm_fn = fmap_files
    with pytest.raises(ValueError, match='coeff_fns is empty'):
        load_fieldmap(coeff_fns, xfm_fn, BOLD_AFFINE, 'j', 0.05)


@pytest.mark.parametrize('pe_direction', [None, '', 'j--', 'y', 'J', '-j', ['j'], 1])
def test_load_fieldmap_bad_pe_direction(fmap_files, pe_direction):
    coeff_fns, xfm_fn = fmap_files
    with pytest.raises(ValueError, match='pe_direction must be one of'):
        load_fieldmap(coeff_fns, xfm_fn, BOLD_AFFINE, pe_direction, 0.05)


@pytest.mark.parametrize('readout_time', [
    0, 0.0, -0.05, np.nan, np.inf, -np.inf, None, '0.05', [0.05], np.array([0.05]),
    np.array(0.05), True, np.bool_(True), 1 + 0j])
def test_load_fieldmap_bad_readout_time(fmap_files, readout_time):
    coeff_fns, xfm_fn = fmap_files
    with pytest.raises(ValueError, match='readout_time must be a finite positive number'):
        load_fieldmap(coeff_fns, xfm_fn, BOLD_AFFINE, 'j', readout_time)


@pytest.mark.parametrize('readout_time', [0.05, 1, np.float64(0.05), np.float32(0.05), np.int64(1)])
def test_load_fieldmap_readout_time_types(fmap_files, readout_time):
    coeff_fns, xfm_fn = fmap_files
    fmap = load_fieldmap(coeff_fns, xfm_fn, BOLD_AFFINE, 'j', readout_time)
    np.testing.assert_allclose(fmap['levels'][0][0], nib.load(coeff_fns[0]).get_fdata() * float(readout_time))


@pytest.mark.parametrize('shape', [(5, 6), (5, 6, 5, 1), (5, 6, 5, 2)])
def test_load_fieldmap_not_3d(fmap_files, tmp_path, shape):
    coeff_fns, xfm_fn = fmap_files
    bad_fn = str(tmp_path / 'bad_coeff.nii.gz')
    nib.save(nib.Nifti1Image(np.zeros(shape), np.eye(4)), bad_fn)
    with pytest.raises(ValueError, match=f'{bad_fn}.* must be 3D'):
        load_fieldmap([coeff_fns[0], bad_fn], xfm_fn, BOLD_AFFINE, 'j', 0.05)


# Integration tests with sdcflows

def test_fieldmap_shift_matches_sdcflows(fmap_files):
    """The shift equals sdcflows' fieldmap times the readout time.

    sdcflows evaluates the B-spline on a grid aligned with the coefficient
    grids (as fMRIPrep does, before resampling it to the BOLD grid), so the
    comparison is on such a grid, mapped into BOLD voxel coordinates.
    """
    transform = pytest.importorskip('sdcflows.transform')
    from scipy.sparse import hstack

    coeff_fns, xfm_fn = fmap_files
    levels = [nib.load(_) for _ in coeff_fns]
    shape = (25, 30, 20)
    target_affine = np.array([  # axis-aligned with a flip, in fieldmap world coordinates
        [-4.0, 0.0, 0.0, 50.0],
        [0.0, 3.5, 0.0, -50.0],
        [0.0, 0.0, 4.0, -40.0],
        [0.0, 0.0, 0.0, 1.0]])
    target = nib.Nifti1Image(np.zeros(shape), target_affine)
    weights = hstack([transform.grid_bspline_weights(target, _, dtype='float64') for _ in levels])
    hz = (weights @ np.concatenate([_.get_fdata().ravel() for _ in levels])).reshape(shape)
    assert np.abs(hz).max() > 1

    fmap = load_fieldmap(coeff_fns, xfm_fn, BOLD_AFFINE, 'j', 0.05)
    # target voxel -> fieldmap world -> BOLD world -> BOLD voxel
    xfm = nt.linear.Affine.from_filename(xfm_fn).matrix  # XFM as stored in the file
    target_to_bold = np.linalg.inv(BOLD_AFFINE) @ xfm @ target_affine
    ijk = np.tensordot(target_to_bold[:3, :3], np.indices(shape), axes=1)
    ijk += target_to_bold[:3, 3].reshape(3, 1, 1, 1)
    np.testing.assert_allclose(fieldmap_shift(fmap, ijk), hz * 0.05, atol=1e-10)

    # Any shape of coordinates; a single point
    pts = ijk.reshape(3, -1)[:, ::7]
    np.testing.assert_allclose(fieldmap_shift(fmap, pts), (hz * 0.05).ravel()[::7], atol=1e-10)
    np.testing.assert_allclose(fieldmap_shift(fmap, ijk[:, 3, 4, 5]), hz[3, 4, 5] * 0.05, atol=1e-10)


@pytest.mark.parametrize('axcodes', ['RAS', 'LAS', 'LPI', 'RPS'])
@pytest.mark.parametrize('pe_direction', PE_DIRECTIONS)
def test_fieldmap_shift_sign_matches_fmriprep(tmp_path, axcodes, pe_direction):
    """Same displacement in world coordinates as fMRIPrep's convention.

    fMRIPrep reorients the BOLD data to positive direction cosines, then uses
    ``-readout_time if (axis_flip ^ pe_flip) else readout_time``
    (``fmriprep.interfaces.resampling.ResampleSeries``).
    """
    tools = pytest.importorskip('sdcflows.utils.tools')
    flips = [code in 'LPI' for code in axcodes]
    bold_affine = np.diag([-2.0 if f else 2.0 for f in flips] + [1.0])
    bold_img = nib.Nifti1Image(np.zeros(BOLD_SHAPE), bold_affine)

    # A constant 10 Hz field around the BOLD data
    coef_affine = np.diag([20.0, 20.0, 20.0, 1.0])
    coef_affine[:3, 3] = -100
    fn = tmp_path / 'coeff.nii.gz'
    nib.save(nib.Nifti1Image(np.full((11, 11, 11), 10.0), coef_affine), fn)
    xfm_fn = tmp_path / 'xfm.txt'
    nt.linear.Affine(np.eye(4)).to_filename(xfm_fn, fmt='itk')
    fmap = load_fieldmap([str(fn)], str(xfm_fn), bold_affine, pe_direction, 0.05)
    pe_axis = fmap['pe_axis']
    center = (np.array(BOLD_SHAPE) - 1) / 2
    shift = fieldmap_shift(fmap, center)
    ours = bold_affine[:3, pe_axis] * shift

    reoriented, codes = tools.ensure_positive_cosines(bold_img)
    axis_flip = codes[pe_axis] in 'LPI'
    pe_flip = pe_direction.endswith('-')
    signed = -0.05 if (axis_flip ^ pe_flip) else 0.05
    theirs = reoriented.affine[:3, pe_axis] * 10.0 * signed

    np.testing.assert_allclose(ours, theirs)
    assert abs(shift) == pytest.approx(0.5)

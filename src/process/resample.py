import numbers

import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
import warnings
import h5py
import nitransforms as nt
from nitransforms.io.itk import ITKCompositeH5

from .fieldmap import fieldmap_shift, load_fieldmap, outside_fieldmap
from .metadata import distortion_parameters


def sample_bold(data, hmc, ijk, fieldmap=None, order=1):
    """Sample BOLD with motion and distortion correction in one interpolation.

    For each volume, transform fixed reference-frame sample positions into
    that volume's voxel coordinates, add the distortion shifts, and sample
    the raw intensities. Average the samples belonging to each location and
    report how often their sampling coordinates fell within the field of view.

    Parameters
    ----------
    data : ndarray or image proxy, shape (I, J, K, T)
        Raw BOLD intensities with time on the last axis and at least one
        volume. Must expose ``.shape`` and support ``data[..., t]`` slicing;
        nested Python lists and NIfTI image objects are not accepted. Pass
        ``img.dataobj`` for a NiBabel proxy or ``np.asanyarray(img.dataobj)``
        for an array. Each volume is converted to float64 when sampled.
        Reading a compressed NIfTI proxy volume by volume can be slow.
    hmc : array_like of shape (T, 4, 4)
        One affine per volume, mapping boldref-frame positions expressed on
        the raw BOLD voxel lattice to voxel coordinates of the moving volume.
        Use identity matrices to omit motion correction. With fMRIPrep's
        world-space matrices ``M`` (``*_desc-hmc_xfm.txt``, which map boldref
        to volume world coordinates), this is ``inv(A) @ M @ A``, where ``A``
        is the affine of the raw data. Matrices are converted to float64;
        their homogeneous last row is expected to be ``[0, 0, 0, 1]``.
    ijk : array_like of shape (3, D, N)
        Boldref-frame sample positions on the raw BOLD voxel lattice, before
        motion or distortion correction. The leading axis contains i, j,
        and k; ``D`` is the number of samples per location (for example,
        cortical depths), and ``N`` is the number of locations. Coordinates
        may be fractional and are converted to float64. ``D`` must be
        positive; use ``D = 1`` to sample individual points without depth
        averaging. ``N = 0`` is allowed and returns empty outputs.
    fieldmap : dict or None, optional
        Output of :func:`process.fieldmap.load_fieldmap` for this run and raw
        image orientation. Default is None, meaning no distortion correction;
        the caller is responsible for deciding whether SDC can be omitted.
    order : int, optional
        Spatial spline interpolation order for BOLD intensities, from 0 to 5.
        Default is 1 (linear); 0 is nearest-neighbor and 3 is cubic, the order
        used by fMRIPrep. Python and NumPy integer scalars are accepted;
        floats and booleans are rejected. This does not change the cubic
        B-spline evaluation of the fieldmap coefficients.

    Returns
    -------
    values : ndarray of float64, shape (T, N)
        Sampled intensities, averaged with equal weights over the ``D``
        samples of each location. All samples contribute, including those
        clipped to the image boundary; coverage is not used as a weight.
    coverage : ndarray of float64, shape (N,)
        Fraction of the ``D * T`` sampling positions inside the field of view,
        measured after motion and distortion correction but before clipping.
        For a spatial axis of length ``n``, the inclusive interval is
        ``[-0.5, n - 0.5]``: the outer faces of its edge voxels. A position
        must satisfy this condition on all three axes. Values below 1 mean
        some samples came from clipped positions and may be inaccurate.

    Raises
    ------
    ValueError
        If ``data`` is not 4D or has no volumes, ``hmc`` has the wrong shape,
        ``ijk`` does not have shape ``(3, D, N)`` with positive ``D``, or
        ``order`` is not an integer scalar from 0 to 5.

    See Also
    --------
    process.fieldmap.load_fieldmap : Prepare a run's fieldmap for sampling.
    process.fieldmap.fieldmap_shift : Evaluate shifts at reference positions.

    Notes
    -----
    For reference position ``p``, the sampling coordinate in volume ``t`` is::

        q = hmc[t, :3, :3] @ p + hmc[t, :3, 3]
        q[fieldmap['pe_axis']] += fieldmap_shift(fieldmap, p)

    The second line is omitted without SDC. The field is evaluated at ``p``,
    before applying motion, and its shift is added afterward along the raw
    voxel phase-encoding axis, matching fMRIPrep's sampling convention. Shifts
    are computed once and reused across volumes. No Jacobian intensity
    modulation is applied.

    Interpolation coordinates are clipped to voxel centers ``[0, n - 1]``
    on each axis. This includes the outer half of an edge voxel even though
    such a position counts as covered. SciPy's ``mode='nearest'`` extends
    intensities by repeating edge voxels for the interpolation boundary
    condition; it does not select nearest-neighbor interpolation unless
    ``order=0``. Orders above 1 use SciPy's spline prefilter for each volume.
    Coverage describes sample positions, not the interpolation kernel's
    support, valid fieldmap coverage, or signal quality.

    Inputs are not modified. Coordinates and motion matrices are expected
    to be finite; their shapes are checked, but finiteness and affine
    validity are not. Nonfinite intensities are not excluded from averaging.
    Image reading, numeric conversion, and fieldmap errors propagate to the
    caller.

    Examples
    --------
    Sample two depths of one location from a volume whose intensity equals i:

    >>> data = np.indices((4, 4, 4), dtype=float)[0][..., None]
    >>> hmc = np.eye(4)[None]
    >>> ijk = np.array([[[1.0], [2.0]],
    ...                 [[1.0], [1.0]],
    ...                 [[1.0], [1.0]]])
    >>> values, coverage = sample_bold(data, hmc, ijk)
    >>> values
    array([[1.5]])
    >>> coverage
    array([1.])
    """
    shape = data.shape
    if len(shape) != 4:
        raise ValueError(f'data must be 4D, but has shape {shape}')
    n_volumes = shape[3]
    if n_volumes == 0:
        raise ValueError(f'data must have at least one volume, but has shape {shape}')
    hmc = np.asarray(hmc, dtype=np.float64)
    if hmc.shape != (n_volumes, 4, 4):
        raise ValueError(f'hmc must have shape ({n_volumes}, 4, 4), got {hmc.shape}')
    ijk = np.asarray(ijk, dtype=np.float64)
    if ijk.ndim != 3 or ijk.shape[0] != 3:
        raise ValueError(f'ijk must have shape (3, D, N), got {ijk.shape}')
    n_samples, n_locations = ijk.shape[1:]
    if n_samples == 0:
        raise ValueError(f'ijk must have at least one sample per location (D > 0), got {ijk.shape}')
    if (isinstance(order, (bool, np.bool_)) or not isinstance(order, numbers.Integral)
            or not 0 <= order <= 5):
        raise ValueError(f'order must be an integer from 0 to 5, got {order!r}')
    points = ijk.reshape(3, -1)
    if fieldmap is not None:
        shift = fieldmap_shift(fieldmap, points)

    last = np.array(shape[:3], dtype=np.float64)[:, np.newaxis] - 1
    values = np.empty((n_volumes, n_locations))
    n_inside = np.zeros(points.shape[1])
    for t in range(n_volumes):
        coords = hmc[t, :3, :3] @ points + hmc[t, :3, 3:]
        if fieldmap is not None:
            coords[fieldmap['pe_axis']] += shift
        n_inside += np.all((coords >= -0.5) & (coords <= last + 0.5), axis=0)
        np.clip(coords, 0, last, out=coords)
        vol = np.asarray(data[..., t], dtype=np.float64)
        sampled = map_coordinates(vol, coords, order=order, mode='nearest')
        values[t] = sampled.reshape(n_samples, n_locations).mean(axis=0)
    coverage = (n_inside / n_volumes).reshape(n_samples, n_locations).mean(axis=0)
    return values, coverage


def resample_run(run, bids_dir, coords, *, order=1, max_outside_fieldmap=0.05):
    """Sample one raw BOLD run at positions given in T1w space.

    Load the run's fMRIPrep transforms and fieldmap, convert the positions
    to raw BOLD voxel coordinates, and sample with :func:`sample_bold`, so
    that motion correction, distortion correction, and coregistration are
    applied in a single interpolation.

    Parameters
    ----------
    run : dict
        One element of the list returned by
        :func:`process.fmriprep.find_runs`, with keys ``'label'``,
        ``'bold_fn'``, ``'hmc_fn'``, ``'coreg_fn'``, and ``'fieldmap'``.
        The raw image must be 4D with at least one volume, and the motion
        file must contain one transform per volume. ``'fieldmap'`` is None
        or a dict with ``'coeff_fns'``, ``'xfm_fn'``, and ``'ref_fn'``;
        ``'ref_fn'`` may be None. Paths may be strings or path-like objects.
    bids_dir : str or os.PathLike
        Root of the raw BIDS dataset, used to read the phase-encoding
        direction and readout time through BIDS metadata inheritance. Only
        read when the run has a fieldmap. Relative paths, including those
        in ``run``, are resolved against the current working directory.
    coords : array_like of shape (D, N, 3)
        World (RAS+, mm) coordinates in fMRIPrep's T1w space: ``D`` samples
        (for example, cortical depths) for each of ``N`` locations. These
        are not T1w voxel indices or FreeSurfer surface RAS coordinates.
        Values are converted to float64 and must be finite; finiteness is
        not checked. ``D`` must be positive; ``N = 0`` is allowed.
    order : int, optional
        Spatial spline interpolation order for BOLD intensities, from 0
        to 5. Default is 1 (linear); 0 is nearest-neighbor and 3 is cubic.
        See :func:`sample_bold` for accepted integer types and boundary
        handling. This does not change the fieldmap's cubic B-splines.
    max_outside_fieldmap : float, optional
        Fraction in [0, 1] above which to emit a UserWarning about positions
        outside the fieldmap grid. Default is 0.05; 0 warns for any outside
        position, and 1 suppresses the warning. The comparison is strictly
        greater than the threshold, which is not validated. The fraction
        counts the ``D * N`` positions before head motion, once per run,
        rather than once per volume. This setting only controls the warning.

    Returns
    -------
    values : ndarray of float64, shape (T, N)
        Sampled intensities for each volume, averaged equally over the
        ``D`` samples per location. Boundary-clipped samples contribute
        too. ``T`` is the number of raw BOLD volumes.
    coverage : ndarray of float64, shape (N,)
        Fraction of the ``D * T`` sampling positions inside the raw BOLD
        field of view after motion and distortion correction, before
        clipping. This does not measure fieldmap coverage or signal quality.
        Both outputs are empty along the location axis when ``N = 0``.

    Raises
    ------
    ValueError
        If ``coords`` does not have shape ``(D, N, 3)`` with positive ``D``,
        or if the image dimensions, motion-array shape, interpolation order,
        or required distortion metadata are invalid.
    numpy.linalg.LinAlgError
        If an affine that must be inverted is singular.

    Warns
    -----
    UserWarning
        If the fraction of positions outside the fieldmap grid exceeds
        ``max_outside_fieldmap``. No check is made without a fieldmap,
        without its ``ref_fn``, or when there are no locations.

    See Also
    --------
    process.fmriprep.find_runs : Discover the run's input paths.
    sample_bold : Sampling, depth averaging, and BOLD coverage conventions.
    process.fieldmap.outside_fieldmap : Check fieldmap grid bounds.

    Notes
    -----
    With ``A`` the affine of the raw BOLD image, ``C`` the matrix of
    ``from-boldref_to-T1w`` (which maps T1w positions to boldref positions),
    and ``M[t]`` the motion matrices (boldref to volume ``t``), positions
    are ``inv(A) @ C @ p`` for a homogeneous column vector ``p`` from
    ``coords``, and motion is ``inv(A) @ M[t] @ A``. The resulting lookup
    positions are in the boldref frame expressed on the raw BOLD voxel
    lattice. Fieldmap shifts are evaluated there before motion and added
    after motion along the raw phase-encoding axis, as in :func:`sample_bold`.

    If ``run['fieldmap']`` is None, no distortion correction is applied.

    Positions outside the fieldmap's field of view (the grid of
    ``run['fieldmap']['ref_fn']``) are still corrected, using the B-spline
    extension of the field. The grid check neither masks samples nor
    changes their shifts; it does not establish that the field estimate is
    reliable inside the grid. No Jacobian intensity modulation is applied.

    The whole run is loaded into memory with NIfTI intensity scaling
    applied. This can promote the stored data type to a floating-point
    type. Each volume is then sampled in float64. Inputs are not modified.
    File-reading, transform-parsing, and metadata errors propagate from
    the loaders; files are still loaded when ``N = 0``.

    Examples
    --------
    Sample one discovered run at prepared T1w-space positions::

        from process.fmriprep import find_runs

        run = find_runs('/data/fmriprep', '/data/bids', '01')[0]
        values, coverage = resample_run(run, '/data/bids', coords, order=1)
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError(f'coords must have shape (D, N, 3), got {coords.shape}')
    img = nib.load(run['bold_fn'])
    affine = img.affine
    inv_affine = np.linalg.inv(affine)

    hmc = nt.linear.LinearTransformsMapping.from_filename(run['hmc_fn'], fmt='itk').matrix
    hmc = inv_affine @ hmc @ affine
    t1w_to_vox = inv_affine @ nt.linear.Affine.from_filename(run['coreg_fn']).matrix
    ijk = np.moveaxis(coords @ t1w_to_vox[:3, :3].T + t1w_to_vox[:3, 3], -1, 0)

    fieldmap = None
    if run['fieldmap'] is not None:
        pe_direction, readout_time = distortion_parameters(run['bold_fn'], bids_dir)
        fieldmap = load_fieldmap(
            run['fieldmap']['coeff_fns'], run['fieldmap']['xfm_fn'],
            affine, pe_direction, readout_time)
        if run['fieldmap']['ref_fn'] is not None and ijk.size:
            outside = outside_fieldmap(
                run['fieldmap']['ref_fn'], run['fieldmap']['xfm_fn'], affine, ijk).mean()
            if outside > max_outside_fieldmap:
                warnings.warn(
                    f"{run['label']}: {outside:.1%} of the sample positions are outside the "
                    "fieldmap's field of view; their distortion correction is extrapolated.")

    return sample_bold(np.asanyarray(img.dataobj), hmc, ijk, fieldmap, order=order)


def interpolate(img, ijk, fill=np.nan, kwargs={'order': 1}):
    shape = img.shape
    ijk = np.moveaxis(ijk, -1, 0)
    if ijk.shape[0] == 4:
        if np.allclose(ijk[3], 1, atol=1e-7):
            # Sometimes t1-to-surface lta not fully orthogonal
            ijk = ijk[:3]
    assert ijk.shape[0] == len(shape), \
        f"Shape of indices {ijk.shape} must match dimensions of input image {shape}."

    if fill is not None:
        invalid = []
        for i, n in enumerate(shape):
            iv = np.logical_or(ijk[i] < -0.5, ijk[i] > n - 0.5)
            invalid.append(iv)
        invalid = np.logical_or.reduce(invalid, axis=0)
        assert invalid.shape == ijk.shape[1:]

    ijk = ijk.copy()
    for i, n in enumerate(shape):
        ijk[i] = np.clip(ijk[i], 0, n - 1)

    if len(ijk.shape) == 1:
        interp = map_coordinates(img, ijk[:, np.newaxis], **kwargs)
        single = True
    else:
        interp = map_coordinates(img, ijk, **kwargs)
        single = False

    if fill is not None:
        if np.any(invalid):
            if not np.can_cast(fill, interp.dtype):
                warnings.warn(
                    "Cannot cast `fill`'s dtype to `interp`'s dtype safely. "
                    f"Casting `interp` from {interp.dtype} to np.float64.")
                interp = interp.astype(np.float64)
            interp[invalid] = fill

    if single:
        interp = interp[0]

    return interp


def compute_warp(xyz1, warp, affine, kwargs={'order': 1}):
    ijk1 = xyz1 @ np.linalg.inv(affine).T
    diff = [interpolate(w.astype(np.float64), ijk1, fill=0., kwargs=kwargs)
            for w in np.moveaxis(warp, -1, 0)]
    diff = np.stack(diff, axis=-1)
    return diff


def parse_warp_image(nifti_fn, to_ras=True):
    """
    References
    ==========
    The data in the DisplacementFieldTransform image is based on the LPS
    coordinates system, and often interpolated linearly:
    https://www.slicer.org/wiki/Coordinate_systems
    https://itk.org/Doxygen/html/classitk_1_1DisplacementFieldTransform.html
    https://simpleitk.org/SPIE2019_COURSE/01_spatial_transformations.html
    """
    img = nib.load(nifti_fn)
    affine = img.affine.copy()
    warp = np.asarray(img.dataobj)
    if len(warp.shape) == 5 and warp.shape[3] == 1:
        warp = warp[:, :, :, 0, :]
    assert len(warp.shape) == 4 and warp.shape[3] == 3
    if to_ras:
        warp *= np.array([-1, -1, 1])
    return warp, affine


def parse_combined_hdf5(h5_fn, to_ras=True):
    h = h5py.File(h5_fn)
    xform = ITKCompositeH5.from_h5obj(h)
    affine = xform[0].to_ras()
    assert h['TransformGroup']['2']['TransformType'][:][0] == b'DisplacementFieldTransform_float_3_3'
    np.testing.assert_array_equal(
        h['TransformGroup']['2']['TransformFixedParameters'][:],
        np.array([193., 229., 193.,  96., 132., -78.,   1.,   1.,   1.,  -1.,   0.,   0.,   0.,  -1.,   0.,
            0.,   0.,   1.]))
    warp = h['TransformGroup']['2']['TransformParameters'][:]
    warp = warp.reshape((193, 229, 193, 3)).transpose(2, 1, 0, 3)
    warp *= np.array([-1, -1, 1])
    warp_affine = np.array(
        [[   1.,    0.,    0.,  -96.],
         [   0.,    1.,    0., -132.],
         [   0.,    0.,    1.,  -78.],
         [   0.,    0.,    0.,    1.]])
    return affine, warp, warp_affine

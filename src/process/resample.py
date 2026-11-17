import numbers
import os

import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
import warnings
import h5py
import nitransforms as nt
from nitransforms.io.itk import ITKCompositeH5

from joblib import Parallel, delayed
from scipy import sparse

from .fieldmap import fieldmap_shift, load_fieldmap, outside_fieldmap
from .fmriprep import find_runs, load_surfaces
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


def check_sdc(run, bids_dir, n_frames=40):
    """Compare sampling with and without SDC against fMRIPrep's reference.

    fMRIPrep's ``desc-coreg_boldref`` image is the run's reference with
    fMRIPrep's own distortion correction applied. The raw data are sampled
    at selected voxel centers on its grid with :func:`sample_bold`, using
    the same volumes and head-motion transforms with and without the
    fieldmap. Correlate each temporal mean with the reference intensities.
    Higher correlation with the fieldmap is evidence of consistent
    susceptibility distortion correction (SDC), not a pass/fail criterion.

    Parameters
    ----------
    run : dict
        One element returned by :func:`process.fmriprep.find_runs`, with
        keys ``'label'``, ``'bold_fn'``, ``'hmc_fn'``, and ``'fieldmap'``.
        ``'fieldmap'`` must be a dict containing ``'coeff_fns'`` and
        ``'xfm_fn'``. The raw image must be 4D with at least one volume,
        with one motion transform per volume. Paths may be strings or
        path-like objects. Neither ``'coreg_fn'`` nor the fieldmap's
        ``'ref_fn'`` is used: sampling takes place in boldref space.
    bids_dir : str or os.PathLike
        Root of the raw BIDS dataset, used to read the phase-encoding
        direction and total readout time through BIDS metadata inheritance.
        Relative paths, including those in ``run``, are resolved against
        the current working directory.
    n_frames : int, optional
        Maximum number of volumes to average; expected to be a positive
        integer. Default is 40. Indices are rounded from evenly spaced
        positions between the first and last volume, using all volumes
        for shorter runs. A value of 1 selects the first volume. Initial
        non-steady-state volumes are not excluded. This limits sampling
        work, but the entire raw image is loaded before selecting volumes.

    Returns
    -------
    result : dict
        Four Python floats. ``'r_sdc'`` and ``'r_nosdc'`` are Pearson
        correlations between the reference and the temporal means with
        and without SDC. Both use the same voxels, whose reference
        intensities are strictly above the 40th percentile of its positive
        intensities. ``'shift_p1'`` and ``'shift_p99'`` are the 1st and 99th
        percentiles of signed fieldmap shifts at those positions, in raw
        BOLD voxels along the phase-encoding axis. Shifts are evaluated
        once, before head motion, rather than separately for each volume.

    Raises
    ------
    ValueError
        If the run has no fieldmap, or required distortion metadata or
        sampling inputs are invalid.
    FileNotFoundError
        If the reference image is missing.

    Notes
    -----
    The reference is ``<name>_desc-coreg_boldref.nii.gz`` in the directory
    of the motion transform, where ``<name>`` is the raw BOLD file name
    without ``_bold.nii[.gz]``. Unlike the transforms, fMRIPrep names this
    image with the raw file's ``part`` entity retained. This lookup is for
    the single-echo runs supported by :func:`process.fmriprep.find_runs`.

    Sampling uses linear interpolation, without Jacobian intensity
    modulation. Out-of-field-of-view samples are clipped to the BOLD image
    boundary and retained in the mean; coverage returned by
    :func:`sample_bold` is not used to exclude voxels. Fieldmap coverage is
    not checked. The intensity threshold is not an anatomical brain mask.

    ``r_sdc > r_nosdc`` is a heuristic expectation, not a guarantee: a
    wrong sign, axis, or readout time usually lowers ``r_sdc``, but a field
    close to zero gives nearly equal correlations, and fMRIPrep's reference
    is itself an estimate. The caller decides what to do with the result.

    Meaningful correlations require at least two selected voxels and
    finite, nonconstant reference and mean intensities. These conditions
    and ``n_frames`` are not explicitly validated. Degenerate inputs can
    raise NumPy errors or produce NaN correlations; nonfinite data are not
    filtered out. Image-reading and transform errors propagate. No files
    are written.

    See Also
    --------
    process.fmriprep.find_runs : Discover the run's input paths.
    sample_bold : Sampling and boundary conventions.
    """
    if run['fieldmap'] is None:
        raise ValueError(f"{run['label']} has no fieldmap, so there is no distortion correction to check")
    name = os.path.basename(run['bold_fn']).rsplit('_bold.nii', 1)[0]
    ref_fn = os.path.join(os.path.dirname(run['hmc_fn']), name + '_desc-coreg_boldref.nii.gz')
    if not os.path.exists(ref_fn):
        raise FileNotFoundError(f"Reference image {ref_fn} of {run['label']} does not exist")
    ref = nib.load(ref_fn)
    reference = np.asarray(ref.dataobj, dtype=np.float64).ravel()
    mask = reference > np.percentile(reference[reference > 0], 40)

    img = nib.load(run['bold_fn'])
    affine = img.affine
    inv_affine = np.linalg.inv(affine)
    frames = np.unique(np.round(np.linspace(0, img.shape[3] - 1, min(n_frames, img.shape[3]))).astype(int))
    data = np.asanyarray(img.dataobj)[..., frames]
    hmc = nt.linear.LinearTransformsMapping.from_filename(run['hmc_fn'], fmt='itk').matrix
    hmc = (inv_affine @ hmc @ affine)[frames]
    # Voxel centers of the reference image (boldref space) on the raw BOLD voxel lattice
    ref_to_vox = inv_affine @ ref.affine
    grid = np.indices(ref.shape[:3]).reshape(3, -1)[:, mask].astype(np.float64)
    ijk = (ref_to_vox[:3, :3] @ grid + ref_to_vox[:3, 3:])[:, np.newaxis]

    pe_direction, readout_time = distortion_parameters(run['bold_fn'], bids_dir)
    fieldmap = load_fieldmap(
        run['fieldmap']['coeff_fns'], run['fieldmap']['xfm_fn'], affine, pe_direction, readout_time)
    result = {}
    for key, fmap in [('r_sdc', fieldmap), ('r_nosdc', None)]:
        values, _ = sample_bold(data, hmc, ijk, fmap)
        result[key] = float(np.corrcoef(values.mean(axis=0), reference[mask])[0, 1])
    shift = fieldmap_shift(fieldmap, ijk)
    result['shift_p1'], result['shift_p99'] = (float(_) for _ in np.percentile(shift, [1, 99]))
    return result


def _resample_surfaces_run(run, bids_dir, coords, n_left_vertices, todo):
    """Sample both hemispheres once and write the missing outputs for a run.

    ``coords`` maps 'l' and 'r' to T1w-space arrays of shape (D, N, 3).
    Concatenate the hemispheres along the vertex axis for one
    :func:`resample_run` call, then split at ``n_left_vertices``.
    ``todo`` maps each hemisphere to (transform, data path, coverage path)
    tuples. Transforms are column-normalized sparse native-to-template
    matrices, or None for native output. Existing files are skipped
    individually. Image loading, transformation, and write errors propagate.
    """
    values, coverage = resample_run(run, bids_dir, np.concatenate([coords[lr] for lr in 'lr'], axis=1))
    for i, lr in enumerate('lr'):
        sl = slice(0, n_left_vertices) if i == 0 else slice(n_left_vertices, None)
        for xform, out_fn, coverage_fn in todo[lr]:
            for data, fn in [(values[:, sl], out_fn), (coverage[sl], coverage_fn)]:
                if os.path.exists(fn):
                    continue
                if xform is not None:
                    data = data @ xform
                os.makedirs(os.path.dirname(fn), exist_ok=True)
                np.save(fn, data)
                print(data.shape, data.dtype, fn)


def resample_surfaces(
        sid, bids_dir, fmriprep_dir, out_dir, xform_dir, coverage_dir, combinations, *,
        fs_dir=None, filter_=None, n_depths=5, n_jobs=1, no_fieldmap=(), fieldmap_prefix=None):
    """Resample a subject's selected BOLD runs and save surface time series.

    Each run is sampled from the raw data at several depths between the
    white and pial surfaces with :func:`resample_run`. Both hemispheres
    are sampled together, with motion correction, distortion correction
    when available, and coregistration applied in one interpolation.
    Samples are averaged equally across depths, then mapped from native
    vertices to the requested template surfaces.

    Parameters
    ----------
    sid : str
        Subject ID without the ``sub-`` prefix.
    bids_dir : str or os.PathLike
        Raw BIDS dataset root, containing ``sub-<sid>``.
    fmriprep_dir : str or os.PathLike
        fMRIPrep derivative dataset root, containing ``sub-<sid>``. Runs
        and surfaces are discovered with :func:`process.fmriprep.find_runs`
        and :func:`process.fmriprep.load_surfaces`.
    out_dir : str or os.PathLike
        Root for sampled BOLD intensities, saved as
        ``<out_dir>/<space>/<l|r>-cerebrum/<tag>/sub-<sid>_<label>.npy``,
        as float64 arrays of shape (T, V): volumes by output vertices.
        ``label`` is the run label returned by ``find_runs``, including
        session and other raw BOLD entities but excluding ``sub-<sid>``.
    xform_dir : str or os.PathLike
        Root for native-to-template overlap matrices, stored at
        ``<xform_dir>/<space>/<sid>_overlap_<l|r>h.npz`` by
        :func:`process.transform.xform_workflow`. Each sparse matrix has
        shape (N, V), with native vertices in the same order as the loaded
        surfaces. Columns are normalized in memory; source files are not
        changed. No matrix is read for the space ``'native'``.
    coverage_dir : str or os.PathLike
        Separate output root for float64 coverage arrays of shape (V,),
        using the same relative paths as the time series. At each native
        vertex, coverage is the fraction of depth-by-volume sampling
        positions inside the raw BOLD field of view after motion and
        distortion correction, before boundary clipping. Template coverage
        is the overlap-weighted average of native coverage. It does not
        measure fieldmap coverage or signal quality; see :func:`sample_bold`.
    combinations : iterable of (str, str)
        Pairs of space and tag, such as
        ``('onavg-ico32', '1step_pial_overlap')``. The only supported tag
        is ``'1step_pial_overlap'``. The space ``'native'`` keeps the
        subject's own vertices.
    fs_dir : str or os.PathLike or None, optional
        The individual subject's FreeSurfer directory, containing ``surf``.
        Used as a fallback when fMRIPrep's GIFTI surfaces are absent;
        incomplete or ambiguous pairs remain errors. Default is None,
        meaning no fallback. See :func:`process.fmriprep.load_surfaces`.
    filter_ : callable or None, optional
        Receives the sorted list of raw BOLD paths once and returns the
        list to process, optionally reordered. This is a list-to-list
        function, not a predicate applied to each file. Default is None,
        selecting all discovered runs; see :func:`process.fmriprep.find_runs`.
    n_depths : int, optional
        Number of sampling depths, at the centers of ``n_depths`` equal
        intervals between white and pial. A positive integer; floats and
        booleans are rejected. Python and NumPy integer scalars are
        accepted. Default is 5, sampling fractions 0.1, 0.3, 0.5, 0.7, and
        0.9; 1 samples the midpoint. White and pial boundaries are excluded.
    n_jobs : int, optional
        Joblib parallelism setting for processing runs. Default is 1
        (sequential). Each worker loads a full raw BOLD run and holds its
        sampled arrays in memory, so memory use grows with concurrency.
    no_fieldmap : collection of str, optional
        Run labels explicitly allowed to lack a fieldmap transform. Default
        is empty. Passed to :func:`process.fmriprep.find_runs`; this cannot
        disable an existing fieldmap transform.
    fieldmap_prefix : mapping of str to str or None, optional
        Run label to coefficient basename prefix before ``_fmapid-``, used
        to select a fieldmap when IDs are reused. Default is None. Passed
        to :func:`process.fmriprep.find_runs`, which documents precedence,
        label validation, and provenance matching.

    Returns
    -------
    None
        Results are written to ``out_dir`` and ``coverage_dir``.

    Raises
    ------
    ValueError
        If a tag is not supported, ``n_depths`` is not a positive integer,
        or run discovery, surface loading, or sampling rejects an input.
    FileNotFoundError
        If an overlap matrix or other required input is missing, or no
        BOLD runs remain after filtering.

    Warns
    -----
    UserWarning
        From :func:`resample_run` when more than 5% of the sampling positions
        across both hemispheres lie outside the available fieldmap grid.

    See Also
    --------
    resample_run : Sample a single run at arbitrary T1w-space positions.
    process.fmriprep.find_runs : Discover runs and resolve fieldmap inputs.
    process.fmriprep.load_surfaces : Load paired surfaces in T1w space.

    Notes
    -----
    This wrapper uses linear BOLD interpolation and the default fieldmap
    warning threshold of :func:`resample_run`. It applies no Jacobian
    intensity modulation. Clipped BOLD samples contribute to the averages;
    coverage is saved separately and is not used as a weight or mask.

    Template-space values are weighted averages of native vertices: each
    column of the transform is normalized to sum to 1. Coverage is mapped
    the same way. Template vertices without overlapping native vertices get
    0 for both.

    Existing files are never rewritten. A run is skipped when all of its
    output files exist; otherwise only the missing files are written, so a
    data file can get its coverage later (or the reverse). Such a file is
    computed with the current code and settings, which its existing
    counterpart may not have been. Existence is the only completion check;
    existing files are not opened or validated. Run discovery and its
    input-path checks still occur before skipping. Surfaces and overlap
    matrices are loaded only if some output is missing; then both
    hemispheres and every requested template space's matrices are loaded.

    All relative directory arguments are resolved against the current
    working directory. Output subdirectories are created as needed.
    Other image-reading, metadata, parallel-execution, and write errors
    propagate. A failure can leave some outputs already written.

    Examples
    --------
    Save native and template time series with separate coverage files::

        resample_surfaces(
            '01', '/data/bids', '/data/fmriprep',
            out_dir='/data/resampled',
            xform_dir='/data/xforms',
            coverage_dir='/data/coverage',
            combinations=[('native', '1step_pial_overlap'),
                          ('onavg-ico32', '1step_pial_overlap')],
            fs_dir='/data/freesurfer/sub-01',
        )
    """
    if (isinstance(n_depths, (bool, np.bool_)) or not isinstance(n_depths, numbers.Integral)
            or n_depths < 1):
        raise ValueError(f'n_depths must be a positive integer, got {n_depths!r}')
    combinations = [tuple(_) for _ in combinations]
    for space, tag in combinations:
        if tag != '1step_pial_overlap':
            raise ValueError(f"Unsupported tag {tag!r} for {space}; only '1step_pial_overlap' is supported")
    runs = find_runs(fmriprep_dir, bids_dir, sid, filter_,
                     no_fieldmap=no_fieldmap, fieldmap_prefix=fieldmap_prefix)

    def out_fns(run, lr, space, tag):
        rel = os.path.join(space, f'{lr}-cerebrum', tag, f"sub-{sid}_{run['label']}.npy")
        return os.path.join(out_dir, rel), os.path.join(coverage_dir, rel)

    def missing(run, lr):
        return [(space, *fns) for space, tag in combinations
                for fns in [out_fns(run, lr, space, tag)]
                if not all(os.path.exists(fn) for fn in fns)]

    runs = [run for run in runs if any(missing(run, lr) for lr in 'lr')]
    if not runs:
        return

    ratios = ((np.arange(n_depths) + 0.5) / n_depths)[:, np.newaxis, np.newaxis]
    coords, xforms = {}, {}
    for lr in 'lr':
        white, pial = load_surfaces(fmriprep_dir, sid, lr, fs_dir=fs_dir)
        coords[lr] = white * ratios + pial * (1 - ratios)
        for space in {space for space, tag in combinations}:
            if space == 'native':
                xforms[lr, space] = None
                continue
            xform_fn = os.path.join(xform_dir, space, f'{sid}_overlap_{lr}h.npz')
            if not os.path.exists(xform_fn):
                raise FileNotFoundError(
                    f'Transform {xform_fn} not found; compute it first (the xform step).')
            xform = sparse.load_npz(xform_fn)
            total = np.asarray(xform.sum(axis=0)).ravel()
            scale = np.divide(1.0, total, out=np.zeros_like(total, dtype=np.float64), where=total > 0)
            xforms[lr, space] = (xform @ sparse.diags(scale)).tocsr()
    n_left_vertices = coords['l'].shape[1]

    jobs = []
    for run in runs:
        todo = {lr: [(xforms[lr, space], out_fn, coverage_fn)
                     for space, out_fn, coverage_fn in missing(run, lr)] for lr in 'lr'}
        jobs.append(delayed(_resample_surfaces_run)(run, bids_dir, coords, n_left_vertices, todo))
    Parallel(n_jobs=n_jobs)(jobs)


def _check_array(fn, shape):
    """Load one NPY array and check its shape, dtype, and finite values.

    ``fn`` is a path; ``shape`` is a tuple of expected axis lengths, with
    None allowing any length on that axis. Rank is always checked. Loading
    disables pickle. Return ``(data, errors)``, where errors are messages
    without a filename prefix. Missing files, files that fail to load for
    any reason, and files that load as something other than an array (an
    NPZ archive) return None in place of data.

    A loaded array is returned even when invalid. Finiteness is checked
    only for float64 arrays; other dtypes already count as an error.
    Value ranges and agreement with a companion array are not checked.
    """
    if not os.path.exists(fn):
        return None, ['missing']
    try:
        data = np.load(fn, allow_pickle=False)
    except Exception as e:
        return None, [f'unreadable ({type(e).__name__}: {e})']
    if not isinstance(data, np.ndarray):
        if hasattr(data, 'close'):
            data.close()
        return None, ['not an NPY array']
    errors = []
    if data.ndim != len(shape) or any(n is not None and n != m for n, m in zip(shape, data.shape)):
        expected = tuple('any' if n is None else n for n in shape)
        errors.append(f'shape {data.shape}, expected {expected}')
    if data.dtype != np.float64:
        errors.append(f'dtype {data.dtype}, expected float64')
    elif not np.all(np.isfinite(data)):
        errors.append(f'{int(np.sum(~np.isfinite(data)))} nonfinite values')
    return data, errors


def check_outputs(
        sid, bids_dir, fmriprep_dir, out_dir, xform_dir, coverage_dir, combinations, *,
        fs_dir=None, filter_=None, no_fieldmap=(), fieldmap_prefix=None,
        low_signal=0.05, low_coverage=0.5, runs=None):
    """Check the files written by :func:`resample_surfaces` for a subject.

    Inspect both time series and coverage for every selected run,
    hemisphere, and space/tag pair. Expected dimensions come from the raw
    BOLD header and surfaces or overlap matrices, independently of the
    saved outputs. This function returns a report without modifying files,
    emitting Python warnings, or running the SDC diagnostic.

    Parameters
    ----------
    sid : str
        Subject ID without the ``sub-`` prefix.
    bids_dir : str or os.PathLike
        Raw BIDS dataset root. BOLD headers provide the expected number
        of volumes, T; raw BOLD intensities are not loaded. Used for run
        discovery only when ``runs`` is None.
    fmriprep_dir : str or os.PathLike
        Derivative dataset root containing ``sub-<sid>``. Used for native
        surface loading and, when ``runs`` is None, run discovery with
        :func:`process.fmriprep.find_runs`, including its checks for required
        transform and fieldmap paths.
    out_dir : str or os.PathLike
        Root of float64 time series with shape (T, V), at
        ``<space>/<l|r>-cerebrum/<tag>/sub-<sid>_<label>.npy`` relative to
        this root. Labels come from ``find_runs``.
    xform_dir : str or os.PathLike
        Root of overlap matrices at ``<space>/<sid>_overlap_<l|r>h.npz``.
        The number of columns gives the expected vertex count, V, for
        each template space. No overlap matrix is read for ``'native'``.
    coverage_dir : str or os.PathLike
        Root of float64 coverage arrays with shape (V,), using the same
        relative paths as the time series.
    combinations : iterable of (str, str)
        Space/tag pairs used for resampling, for example
        ``('onavg-ico32', '1step_pial_overlap')``. For space ``'native'``,
        V comes from :func:`process.fmriprep.load_surfaces`. Tags select
        output subdirectories; this function does not validate them.
    fs_dir : str or os.PathLike or None, optional
        Individual subject's FreeSurfer directory, containing ``surf``.
        Fallback for native vertex counts when fMRIPrep GIFTI surfaces
        are absent. Default is None, meaning no fallback.
    filter_ : callable or None, optional
        Receives the sorted list of raw BOLD paths once and returns the
        list to check. Default is None, selecting all discovered runs.
    no_fieldmap : collection of str, optional
        Run labels explicitly allowed to lack a fieldmap transform.
        Default is empty. Passed to ``find_runs``; it does not disable
        an existing transform.
    fieldmap_prefix : mapping of str to str or None, optional
        Run label to coefficient basename prefix before ``_fmapid-``,
        passed to ``find_runs`` to resolve reused fieldmap IDs. Default
        is None. Use the same overrides as for resampling.
    low_signal : float, optional
        Flag vertices whose temporal mean is strictly below this fraction
        of the median temporal mean across the file's vertices. Default
        is 0.05. This relative heuristic assumes positive BOLD intensities;
        it does not flag an entirely zero file, whose median is also zero.
    low_coverage : float, optional
        Flag vertices whose coverage is strictly below this threshold.
        Default is 0.5. Native coverage is the fraction of depth-by-volume
        positions within the raw BOLD field of view after motion and
        distortion correction. Template coverage is its overlap-weighted
        average; vertices without native overlap have coverage zero.
    runs : list of dict or None, optional
        Runs already selected with ``find_runs``, in the order to check.
        Only ``'label'`` and ``'bold_fn'`` are used. They must belong to
        ``sid`` and have unique labels. Default is None, which discovers
        runs here. Any supplied list, including an empty one, bypasses
        discovery: ``filter_`` is not called, ``no_fieldmap`` and
        ``fieldmap_prefix`` are ignored, and derivative transform and
        fieldmap paths are not revalidated. This lets a caller share one
        selection between output checks and other diagnostics.

    Returns
    -------
    report : dict
        ``'errors'`` and ``'warnings'`` contain lists of strings. Output
        messages start with the relative NPY path; coverage errors also
        include ``'coverage:'``. Input errors instead identify the run
        label or space and hemisphere. Warning messages give vertex counts
        and are included only when those counts are positive.

        ``'runs'`` maps each run label to a dictionary with ``'n_frames'``
        (an int, or None if the raw header could not be read) and
        ``'outputs'``. The latter maps relative NPY paths to summaries with
        ``'n_vertices'`` (int, from the saved time series), ``'median_mean'``
        (float), ``'n_low_signal'`` (int), and ``'n_low_coverage'`` (int).
        The median is defined as zero when there are no vertices.

        A summary and its warnings are omitted if either member of the
        time-series/coverage pair has errors. Summaries can still be
        present when an input dimension is unknown; consult ``'errors'``
        before treating them as validated outputs.

    Raises
    ------
    ValueError, FileNotFoundError
        Run-discovery failures propagate before any outputs are checked
        when ``runs`` is None. Reader failures during file validation are
        recorded in the report instead. Invalid caller-supplied structures
        and unexpected exceptions outside file reading can still propagate.

    See Also
    --------
    resample_surfaces : Write the paired time-series and coverage arrays.
    check_sdc : Compare sampling with and without distortion correction.

    Notes
    -----
    File checks report missing or unreadable arrays, incorrect rank or
    known axis lengths, dtype other than float64, and nonfinite float64
    values. Coverage must lie within [0, 1], with an absolute tolerance
    of 1e-6. Any failure to determine T or V is recorded as an input
    error, and checking continues without that axis-length expectation.
    Rank, dtype, and value checks still apply.

    Low-signal and low-coverage warnings are diagnostics, not proof of an
    incorrect output. Coverage does not measure fieldmap support or signal
    quality. Boundary-clipped samples contribute to the saved intensities,
    and zero-overlap template vertices have both zero signal and coverage.
    Thresholds are used as supplied, without range or finiteness validation.

    Arrays are loaded in full, one pair at a time. All relative directory
    arguments are resolved against the current working directory. These
    checks do not establish that existing outputs were produced with the
    current inputs or resampling settings.
    """
    combinations = [tuple(_) for _ in combinations]
    if runs is None:
        runs = find_runs(fmriprep_dir, bids_dir, sid, filter_,
                         no_fieldmap=no_fieldmap, fieldmap_prefix=fieldmap_prefix)
    errors, warns, report = [], [], {}

    n_vertices = {}
    for lr in 'lr':
        for space in sorted({space for space, tag in combinations}):
            try:
                if space == 'native':
                    n = load_surfaces(fmriprep_dir, sid, lr, fs_dir=fs_dir)[0].shape[0]
                else:
                    n = sparse.load_npz(os.path.join(xform_dir, space, f'{sid}_overlap_{lr}h.npz')).shape[1]
            except Exception as e:
                errors.append(f'{space} {lr}h: cannot get the number of vertices '
                              f'({type(e).__name__}: {e})')
                n = None
            n_vertices[lr, space] = n

    for run in runs:
        label = run['label']
        try:
            n_frames = int(nib.load(run['bold_fn']).shape[3])
        except Exception as e:
            errors.append(f'{label}: cannot get the number of volumes ({type(e).__name__}: {e})')
            n_frames = None
        outputs = {}
        for lr in 'lr':
            for space, tag in combinations:
                rel = os.path.join(space, f'{lr}-cerebrum', tag, f'sub-{sid}_{label}.npy')
                nv = n_vertices[lr, space]
                data, errs = _check_array(os.path.join(out_dir, rel), (n_frames, nv))
                coverage, cov_errs = _check_array(os.path.join(coverage_dir, rel), (nv,))
                if coverage is not None and not cov_errs:
                    if coverage.min(initial=0.0) < -1e-6 or coverage.max(initial=0.0) > 1 + 1e-6:
                        cov_errs.append(f'values outside [0, 1] ({coverage.min()} to {coverage.max()})')
                errors += [f'{rel}: {_}' for _ in errs] + [f'{rel}: coverage: {_}' for _ in cov_errs]
                if errs or cov_errs:
                    continue
                mean = data.mean(axis=0)
                median = float(np.median(mean)) if mean.size else 0.0
                n_low = int(np.sum(mean < low_signal * median))
                n_cov = int(np.sum(coverage < low_coverage))
                if n_low:
                    warns.append(f'{rel}: {n_low} vertices with low signal')
                if n_cov:
                    warns.append(f'{rel}: {n_cov} vertices with low coverage')
                outputs[rel] = {'n_vertices': int(data.shape[1]), 'median_mean': median,
                                'n_low_signal': n_low, 'n_low_coverage': n_cov}
        report[label] = {'n_frames': n_frames, 'outputs': outputs}
    return {'runs': report, 'errors': errors, 'warnings': warns}


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

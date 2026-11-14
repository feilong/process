import json
import numbers
import os
from glob import glob

import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
import warnings
import h5py
from nitransforms.io.itk import ITKCompositeH5

from .fieldmap import fieldmap_shift


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


def list_bold_runs(bids_dir, sid, filter_=None):
    """List a subject's raw BOLD files, optionally selecting a subset.

    Parameters
    ----------
    bids_dir : str or os.PathLike
        Raw BIDS dataset root, containing ``sub-<sid>``. Relative paths are
        resolved against the current working directory.
    sid : str
        Subject ID, without ``sub-``.
    filter_ : callable or None, optional
        Called once with the sorted list of path strings. Must return a
        list containing the paths to keep; it may also reorder them. This
        is a list-to-list function, not a predicate called for each file.

    Returns
    -------
    bold_fns : list of str
        Paths matching ``sub-<sid>/func/*_bold.nii[.gz]`` or
        ``sub-<sid>/ses-*/func/*_bold.nii[.gz]``, sorted lexicographically
        before filtering. Paths retain the supplied root's relative or
        absolute form. An empty list is allowed, including when the dataset
        or subject directory does not exist.

    Notes
    -----
    Only filenames are inspected. Images and metadata are not opened, and
    BIDS validity is not checked. Files with ``echo`` or ``chunk`` entities
    are listed too; :func:`find_runs` rejects them if they are selected.

    Examples
    --------
    Select resting-state runs (the dataset must exist)::

        bold_fns = list_bold_runs(
            '/data/bids', '01',
            filter_=lambda fns: [fn for fn in fns
                                 if '_task-rest_' in os.path.basename(fn)],
        )
    """
    bold_fns = []
    for pattern in ['func', os.path.join('ses-*', 'func')]:
        for ext in ['.nii', '.nii.gz']:
            bold_fns += glob(os.path.join(bids_dir, f'sub-{sid}', pattern, f'*_bold{ext}'))
    bold_fns = sorted(bold_fns)
    if filter_ is not None:
        bold_fns = filter_(bold_fns)
    return bold_fns


# The entities that fMRIPrep keeps in the names of functional transforms: the
# path pattern for `func/*_from-*_to-*_xfm` in niworkflows' `nipreps.json`
# (1.12.2 to 1.15.4) retains these raw BOLD entities, so `echo` and `part` are
# dropped from all transforms (head motion included). `chunk` is not an entity
# in niworkflows, which moves it next to the suffix (`..._desc-hmc_chunk-1_xfm.txt`);
# it is not supported here.
XFM_ENTITIES = ('sub', 'ses', 'task', 'acq', 'ce', 'rec', 'dir', 'run')


def _coefficient_files(fmriprep_dir, fmap_dirs, sid, fmapid, xfm_fn, label, prefix=None):
    """Resolve one coefficient group for a run's boldref-to-fieldmap transform.

    Search ``fmap_dirs`` for compressed coefficient images of subject
    ``sid`` with the exact filename ID ``fmapid`` (already sanitized by
    fMRIPrep). Group matches by directory and basename before ``_fmapid-``;
    the remaining ``desc-coeff*`` files are spline levels of that group.
    Return the selected group's lexicographically sorted path strings.

    A group's session belongs to its source images, which need not share
    the BOLD run's session. Separately processed sessions can also reuse
    automatically assigned fieldmap IDs. Select a group in this order:

    1. the unique group with the explicit basename ``prefix``, if supplied;
    2. the only group with this ID;
    3. the only group that the transform's ``Sources`` (in its JSON sidecar,
       written by fMRIPrep 25.2) points to.

    ``fmriprep_dir`` anchors the output-relative paths used to match
    provenance. A ``bids::<path>`` source is interpreted as a path within
    that output dataset; absolute paths containing the same directory and
    fieldmap basename prefix can also match. No session-based preference
    is applied. An explicit prefix takes precedence over provenance, and
    a single group is accepted without reading the sidecar.

    Raise ``FileNotFoundError`` if no coefficients match the ID, or
    ``ValueError`` if the prefix or provenance does not select exactly one
    group. ``label`` identifies the BOLD run in these errors. Errors reading
    or decoding a needed JSON sidecar propagate. Coefficient images are
    not opened or validated here.
    """
    groups = {}
    for d in fmap_dirs:
        for fn in sorted(glob(os.path.join(d, f'sub-{sid}_*fmapid-{fmapid}_desc-coeff*_fieldmap.nii.gz'))):
            groups.setdefault((d, os.path.basename(fn).split('_fmapid-')[0]), []).append(fn)
    if not groups:
        raise FileNotFoundError(
            f'No coefficient files for fieldmap {fmapid} of {label} in {sorted(set(fmap_dirs))}')
    prefixes = sorted(p for _, p in groups)
    if prefix is not None:
        matches = [fns for (_, p), fns in groups.items() if p == prefix]
        if len(matches) != 1:
            raise ValueError(
                f'fieldmap_prefix {prefix!r} of {label} matches {len(matches)} sets of coefficient '
                f'files for fieldmap {fmapid}; prefixes: {prefixes}')
        return matches[0]
    if len(groups) == 1:
        return next(iter(groups.values()))

    sidecar = xfm_fn[:-len('.txt')] + '.json'
    sources = []
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            sources = json.load(f).get('Sources') or []
    # Sources are BIDS URIs (`bids::<path in fMRIPrep's output>`) or paths
    sources = ['/' + _[len('bids::'):] if _.startswith('bids::') else _ for _ in sources]
    matches = []
    for (d, p), fns in groups.items():
        target = '/' + os.path.join(os.path.relpath(d, fmriprep_dir), f'{p}_fmapid-{fmapid}_')
        if any(target in src for src in sources):
            matches.append(fns)
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f'Several sets of coefficient files for fieldmap {fmapid} of {label} (prefixes {prefixes}), '
        f'and the provenance of {os.path.basename(xfm_fn)} does not identify one; '
        f'pass fieldmap_prefix={{{label!r}: <one of the prefixes>}}')


def find_runs(
    fmriprep_dir, bids_dir, sid, filter_=None, *, no_fieldmap=(), fieldmap_prefix=None,
):
    """Find raw BOLD runs and their fMRIPrep 25.x resampling inputs.

    Runs are the raw BOLD files in the BIDS dataset (:func:`list_bold_runs`).
    Transform paths follow fMRIPrep 25.1/25.2 naming and require the files
    saved at ``--level minimal`` or higher. All selected runs must have
    motion and coregistration transforms. Use ``filter_`` to exclude runs
    that were not processed or are not supported by this finder.

    If any coefficient files exist for the subject, each selected run must
    also have a boldref-to-fieldmap transform or be listed in
    ``no_fieldmap``. A fieldmap can come from another session; its source
    is resolved by ID, provenance, or an explicit ``fieldmap_prefix``.

    Parameters
    ----------
    fmriprep_dir : str or os.PathLike
        fMRIPrep derivative dataset root, containing ``sub-<sid>``.
    bids_dir : str or os.PathLike
        Raw BIDS dataset root, containing ``sub-<sid>``. Both directory
        arguments are resolved against the current working directory if
        relative; neither is interpreted relative to the other.
    sid : str
        Subject ID, without ``sub-``.
    filter_ : callable or None, optional
        Receives the sorted list from :func:`list_bold_runs` and returns a
        list of paths to keep, in the desired order. It is called once,
        not once per file. With None, all discovered files are selected.
    no_fieldmap : collection of str, optional
        Run labels explicitly allowed to lack a fieldmap transform, e.g.,
        ``{'ses-1_task-wd_run-01'}``. Labels omit ``sub-<sid>_`` and
        ``_bold.nii[.gz]``, retaining all other entities and zero padding.
        This acknowledges missing SDC; it cannot disable an existing
        fieldmap transform, which instead causes an error.
    fieldmap_prefix : mapping of str to str or None, optional
        Run label to coefficient basename prefix before ``_fmapid-``,
        including every entity in that prefix. For example,
        ``{'ses-2_task-x': 'sub-01_ses-1_acq-b'}`` selects coefficients
        named ``sub-01_ses-1_acq-b_fmapid-<id>_desc-coeff*_fieldmap.nii.gz``.
        The ID still comes from the run's transform. An explicit prefix
        must match exactly one group and takes precedence over provenance.
        Use it when repeated IDs cannot be resolved from the transform's
        JSON ``Sources``, notably with fMRIPrep 25.1 outputs.

        Keys in both override arguments are checked against all discovered
        raw runs before filtering. Entries for existing but filtered-out
        runs are allowed; their values and output files are not checked.

    Returns
    -------
    runs : list of dict
        One per selected run, in filter order (lexicographic path order
        without a filter). Paths are strings, retaining their respective
        input root's relative or absolute form. Each dictionary contains:

        - ``'label'``: the run's name without ``sub-<sid>_``, e.g.,
          ``'ses-1_task-rest_run-01'``, also without ``_bold.nii[.gz]``;
        - ``'bold_fn'``: the raw BOLD file;
        - ``'hmc_fn'``, ``'coreg_fn'``: the head-motion and boldref-to-T1w
          transforms;
        - ``'fieldmap'``: None (no distortion correction), or a dict with
          ``'xfm_fn'`` (boldref-to-fieldmap transform) and ``'coeff_fns'``
          (a sorted list of B-spline coefficient paths from one group).

    Raises
    ------
    FileNotFoundError
        If no runs remain after filtering, a required motion or
        coregistration transform is missing, or a fieldmap transform has
        no matching coefficients.
    ValueError
        If selected files contain ``echo`` or ``chunk`` entities, share a
        transform basename, or have several candidate fieldmap transforms;
        if coefficients cannot be resolved to one group; or if an override
        names an unknown run, contradicts its available fieldmap transform,
        or fails to select a unique coefficient group. Also raised when
        coefficients exist for the subject but a selected run has neither
        a fieldmap transform nor a ``no_fieldmap`` entry.

    See Also
    --------
    list_bold_runs : List raw files without checking derivative inputs.
    process.fieldmap.load_fieldmap : Load coefficients as voxel shifts.
    sample_bold : Sample image data using loaded transforms and shifts.

    Notes
    -----
    All three transform basenames retain the raw BOLD entities in
    ``XFM_ENTITIES``: ``sub``, ``ses``, ``task``, ``acq``, ``ce``, ``rec``,
    ``dir``, and ``run``. In particular, ``part`` is dropped from motion
    transforms too. Filter out unwanted parts when multiple raw files
    would otherwise share transforms. Any ``echo`` entity is rejected,
    even if only one echo file is selected.

    Coefficients are searched in both ``sub-<sid>/fmap`` and
    ``sub-<sid>/ses-*/fmap`` and grouped by directory and basename prefix
    before ``_fmapid-``. Selection uses an explicit prefix first, otherwise
    the only group with the transform's ID, otherwise the unique group
    identified by the transform's JSON ``Sources``. There is no preference
    for the BOLD run's session. fMRIPrep 25.2 can identify its output
    fieldmap reference with a ``bids::`` URI or an absolute path; 25.1's
    raw-BOLD-only provenance does not distinguish repeated fieldmap IDs.
    Missing or inconclusive provenance with several groups is an error.

    The finder checks paths and reads provenance only when needed. It does
    not load images or transform matrices, infer associations from raw
    metadata, validate file contents, or detect the installed fMRIPrep
    version. If neither coefficients nor a fieldmap transform exist, the
    run is returned with ``fieldmap=None``. Errors reading a needed JSON
    sidecar propagate.

    The returned paths still need loading before :func:`sample_bold` can
    use them. :func:`process.fieldmap.load_fieldmap` also needs the raw
    BOLD affine, phase-encoding direction, and readout time; these are not
    provided by this finder.

    Examples
    --------
    Acknowledge a run that fMRIPrep processed without SDC::

        runs = find_runs(
            '/data/fmriprep', '/data/bids', '01',
            no_fieldmap={'task-wd_run-01'},
        )

    Resolve a reused fieldmap ID when provenance is insufficient. Labels
    and prefixes must match the actual dataset::

        runs = find_runs(
            '/data/fmriprep', '/data/bids', '01',
            fieldmap_prefix={
                'ses-1_task-rest': 'sub-01_ses-1',
                'ses-2_task-rest': 'sub-01_ses-2',
            },
        )
    """
    all_fns = list_bold_runs(bids_dir, sid)
    labels = {os.path.basename(fn).split('_bold.nii')[0][len(f'sub-{sid}_'):] for fn in all_fns}
    fieldmap_prefix = fieldmap_prefix or {}
    for arg, value in [('no_fieldmap', no_fieldmap), ('fieldmap_prefix', fieldmap_prefix)]:
        unknown = set(value) - labels
        if unknown:
            raise ValueError(f'{arg} has labels of runs that do not exist: {sorted(unknown)}')
    bold_fns = all_fns if filter_ is None else filter_(all_fns)
    if not bold_fns:
        raise FileNotFoundError(f'No BOLD runs of sub-{sid} in {bids_dir}')

    subj_dir = os.path.join(fmriprep_dir, f'sub-{sid}')
    fmap_dirs = [os.path.join(subj_dir, 'fmap')] + sorted(glob(os.path.join(subj_dir, 'ses-*', 'fmap')))
    has_fieldmaps = any(glob(os.path.join(d, '*_desc-coeff*_fieldmap.nii.gz')) for d in fmap_dirs)

    runs = []
    xfm_bases = {}
    for bold_fn in bold_fns:
        name = os.path.basename(bold_fn).split('_bold.nii')[0]
        label = name[len(f'sub-{sid}_'):]
        if '_echo-' in f'_{label}':
            raise ValueError(f'{bold_fn} is multi-echo, which is not supported yet')
        if '_chunk-' in f'_{label}':
            raise ValueError(f'{bold_fn} has a chunk entity, which is not supported')
        rel_dir = os.path.relpath(os.path.dirname(bold_fn), bids_dir)
        xfm_name = '_'.join(_ for _ in name.split('_') if _.split('-', 1)[0] in XFM_ENTITIES)
        xfm_base = os.path.join(fmriprep_dir, rel_dir, xfm_name)
        if xfm_base in xfm_bases:
            raise ValueError(
                f'{label} and {xfm_bases[xfm_base]} have the same fMRIPrep transforms ({xfm_name}_*); '
                'leave one out with filter_')
        xfm_bases[xfm_base] = label
        hmc_fn = xfm_base + '_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt'
        coreg_fn = xfm_base + '_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt'
        for fn in [hmc_fn, coreg_fn]:
            if not os.path.exists(fn):
                raise FileNotFoundError(f'fMRIPrep output {fn} of {bold_fn} does not exist')
        run = {'label': label, 'bold_fn': bold_fn, 'hmc_fn': hmc_fn, 'coreg_fn': coreg_fn,
               'fieldmap': None}
        # 25.1: `..._from-boldref_to-<fmapid>_mode-image_xfm.txt`; 25.2 adds `desc-fmap`
        xfm_fns = [_ for _ in sorted(glob(xfm_base + '_from-boldref_to-*_mode-image*_xfm.txt'))
                   if _ != coreg_fn]
        if len(xfm_fns) > 1:
            raise ValueError(f'Several boldref-to-fieldmap transforms for {label}: {xfm_fns}')
        if xfm_fns:
            if label in no_fieldmap:
                raise ValueError(f'{label} is listed in no_fieldmap, but has a fieldmap: {xfm_fns[0]}')
            fmapid = os.path.basename(xfm_fns[0])[len(xfm_name + '_from-boldref_to-'):]
            fmapid = fmapid.split('_mode-image')[0]
            coeff_fns = _coefficient_files(fmriprep_dir, fmap_dirs, sid, fmapid, xfm_fns[0], label,
                                           fieldmap_prefix.get(label))
            run['fieldmap'] = {'xfm_fn': xfm_fns[0], 'coeff_fns': coeff_fns}
        elif label in fieldmap_prefix:
            raise ValueError(f'{label} is in fieldmap_prefix, but has no boldref-to-fieldmap transform')
        elif has_fieldmaps and label not in no_fieldmap:
            raise ValueError(
                f'sub-{sid} has fieldmaps, but {label} has no boldref-to-fieldmap transform; '
                'list it in no_fieldmap if fMRIPrep processed it without one')
        runs.append(run)
    return runs


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

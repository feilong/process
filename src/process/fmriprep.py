import json
import os
from glob import glob

import nibabel as nib
import nitransforms as nt
import numpy as np

from .bids import list_bold_runs


def fmriprep_cmd(config, *additional_options):
    cmd = [
        'singularity', 'run', '-e',
        *config['singularity_options'], '-H', config['singularity_home'],
        config['singularity_image'],
        '--participant-label', config['sid'],
        *config['fmriprep_options'],
        *additional_options,
        '--fs-license-file',  config['singularity_home'] + '/FS_license.txt',
        '-w', config['fmriprep_work'],
        config['bids_dir'], config['fmriprep_out'], 'participant',
    ]
    return cmd


def fmriprep_success(returncode, stdout_fn, fmriprep_out):
    if returncode == 0:
        return True

    if os.path.exists(stdout_fn):
        with open(stdout_fn, 'r') as f:
            content = f.read()
        if 'fMRIPrep finished successfully!' in content:
            return True

        crashes = glob(os.path.join(fmriprep_out, 'log', '*', 'crash-*.txt'))
        if len(crashes) <= 2 and all(['_midthickness' in _ for _ in crashes]):
            return True

    return False


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

    Runs are the raw BOLD files in the BIDS dataset (:func:`process.bids.list_bold_runs`).
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
          ``'xfm_fn'`` (boldref-to-fieldmap transform), ``'coeff_fns'``
          (a sorted list of B-spline coefficient paths from one group),
          and ``'ref_fn'`` (the group's ``desc-preproc_fieldmap`` image,
          used to check fieldmap grid bounds; None if missing). A None
          ``'ref_fn'`` skips that check but does not disable SDC.

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
    process.resample.resample_run : Load a run and sample T1w-space positions.
    process.fieldmap.load_fieldmap : Load coefficients as voxel shifts.
    process.resample.sample_bold : Sample image data using loaded transforms and shifts.

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

    The returned paths still need loading before
    :func:`process.resample.sample_bold` can use them.
    :func:`process.fieldmap.load_fieldmap` also needs the raw BOLD affine,
    phase-encoding direction, and readout time; these are not provided by
    this finder.

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
            ref_fn = coeff_fns[0].rsplit('_desc-coeff', 1)[0] + '_desc-preproc_fieldmap.nii.gz'
            run['fieldmap'] = {'xfm_fn': xfm_fns[0], 'coeff_fns': coeff_fns,
                               'ref_fn': ref_fn if os.path.exists(ref_fn) else None}
        elif label in fieldmap_prefix:
            raise ValueError(f'{label} is in fieldmap_prefix, but has no boldref-to-fieldmap transform')
        elif has_fieldmaps and label not in no_fieldmap:
            raise ValueError(
                f'sub-{sid} has fieldmaps, but {label} has no boldref-to-fieldmap transform; '
                'list it in no_fieldmap if fMRIPrep processed it without one')
        runs.append(run)
    return runs


def load_surfaces(fmriprep_dir, sid, lr, *, fs_dir=None):
    """Load the white and pial surfaces of a hemisphere in T1w space.

    Prefer one complete pair of fMRIPrep GIFTI surfaces, written at
    ``--level full``. When neither surface is found for this hemisphere,
    optionally load the native FreeSurfer pair and map it to T1w space
    with fMRIPrep's T1w-to-fsnative transform. Incomplete or ambiguous
    fMRIPrep outputs raise an error instead of triggering the fallback.

    Parameters
    ----------
    fmriprep_dir : str or os.PathLike
        fMRIPrep derivative dataset root, containing ``sub-<sid>``.
    sid : str
        Subject ID without the ``sub-`` prefix.
    lr : {'l', 'r'}
        Hemisphere, using a lowercase letter. Uppercase values are rejected.
    fs_dir : str or os.PathLike or None, optional
        The individual subject's FreeSurfer directory, containing ``surf``;
        do not pass the parent ``SUBJECTS_DIR``. Must be the reconstruction
        associated with fMRIPrep's T1w-to-fsnative transform. Only read when
        neither fMRIPrep surface exists for the requested hemisphere.
        Default is None, meaning no fallback. Both directory arguments are
        resolved against the current working directory if relative.

    Returns
    -------
    white, pial : ndarray of float64, shape (N, 3)
        White and pial vertex positions in that order, in fMRIPrep's T1w
        world space (RAS+, mm), not voxel indices or FreeSurfer surface
        RAS coordinates. Vertex order is preserved from the native meshes,
        and both arrays have the same shape. Triangle connectivity is not
        returned. To use :func:`process.resample.resample_run`, first form
        sample positions of shape ``(D, N, 3)`` between these surfaces.

    Raises
    ------
    FileNotFoundError
        If neither fMRIPrep surface exists and ``fs_dir`` is None, or a
        required fallback transform or FreeSurfer surface file is missing.
    ValueError
        If ``lr`` is invalid, a white or pial surface has no counterpart
        with the same directory and filename prefix, several pairs or
        transforms match, or the loaded coordinate arrays differ in shape.

    See Also
    --------
    find_runs : Discover raw BOLD runs and their fMRIPrep transforms.
    process.resample.resample_run : Sample a run at T1w-space positions.

    Notes
    -----
    Files are searched in ``sub-<sid>/anat`` and ``sub-<sid>/ses-*/anat``:
    ``*_hemi-{L,R}_{white,pial}.surf.gii``, and for the fallback
    ``*_from-T1w_to-fsnative_mode-image_xfm.txt``. White and pial must share
    a directory and the entire filename prefix before ``_white.surf.gii``
    or ``_pial.surf.gii``. Every matching surface must have its counterpart,
    and there must be exactly one pair across all searched directories.
    A subject-level pair is not preferred over a session-level pair.
    Separately processed sessions with multiple pairs are therefore
    ambiguous; this function does not select an anatomical session.

    GIFTI pointset arrays are already in T1w world coordinates and are read
    directly. No additional transform is applied to them. For the fallback,
    ``surf/<lr>h.white`` and ``surf/<lr>h.pial`` are read with their volume
    metadata. Each surface's ``cras`` offset is added to its coordinates,
    then the RAS+ matrix loaded by NiTransforms maps fsnative world
    positions into T1w space. Despite the filename's ``from-T1w_to-fsnative``
    direction, this is a pull transform; its matrix is used without
    inversion. Exactly one matching transform is required for the fallback.

    Inputs are expected to be fMRIPrep outputs and, for the fallback, the
    corresponding FreeSurfer reconstruction. Matching array shapes do not
    verify topology, vertex correspondence, or provenance. Coordinate
    finiteness is not checked. Missing ``cras`` metadata raises KeyError;
    other file-reading and transform-parsing errors propagate from NiBabel
    and NiTransforms. No files are modified.

    Examples
    --------
    Load the left hemisphere and form five equally spaced cortical depths,
    including the white and pial boundaries (the dataset must exist)::

        white, pial = load_surfaces(
            '/data/fmriprep', '01', 'l',
            fs_dir='/data/freesurfer/sub-01',
        )
        depths = np.linspace(0, 1, 5)[:, None, None]
        coords = (1 - depths) * white[None] + depths * pial[None]
    """
    if lr not in ('l', 'r'):
        raise ValueError(f"lr must be 'l' or 'r', got {lr!r}")
    sub_dir = os.path.join(fmriprep_dir, f'sub-{sid}')
    anat_dirs = [os.path.join(sub_dir, 'anat')] + sorted(glob(os.path.join(sub_dir, 'ses-*', 'anat')))

    def find(name):
        fns = [fn for d in anat_dirs for fn in sorted(glob(os.path.join(d, f'sub-{sid}_*{name}')))]
        if len(fns) > 1:
            raise ValueError(f'Several files match *{name} for sub-{sid}: {fns}')
        return fns[0] if fns else None

    kinds = ('white', 'pial')
    groups = {}  # one per directory and filename prefix, so that the pair shares a T1w frame
    for d in anat_dirs:
        for kind in kinds:
            suffix = f'_{kind}.surf.gii'
            for fn in sorted(glob(os.path.join(d, f'sub-{sid}_*hemi-{lr.upper()}{suffix}'))):
                groups.setdefault(fn[:-len(suffix)], {})[kind] = fn
    incomplete = [list(group.values()) for group in groups.values() if len(group) < len(kinds)]
    if incomplete:
        raise ValueError(f'White or pial surface without its counterpart for sub-{sid}: {incomplete}')
    if len(groups) > 1:
        raise ValueError(f'Several pairs of white and pial surfaces for sub-{sid}: {sorted(groups)}')
    if groups:
        [group] = groups.values()
        surfaces = [nib.load(group[kind]).agg_data('pointset').astype(np.float64) for kind in kinds]
    elif fs_dir is None:
        raise FileNotFoundError(
            f'No hemi-{lr.upper()} white and pial surfaces for sub-{sid} in {anat_dirs}; '
            'run fMRIPrep with --level full, or pass fs_dir to use the FreeSurfer surfaces.')
    else:
        xfm_fn = find('from-T1w_to-fsnative_mode-image_xfm.txt')
        if xfm_fn is None:
            raise FileNotFoundError(f'No T1w-to-fsnative transform for sub-{sid} in {anat_dirs}')
        to_t1w = nt.linear.Affine.from_filename(xfm_fn).matrix
        surfaces = []
        for kind in kinds:
            coords, _, info = nib.freesurfer.read_geometry(
                os.path.join(fs_dir, 'surf', f'{lr}h.{kind}'), read_metadata=True)
            coords = coords.astype(np.float64) + info['cras']
            surfaces.append(coords @ to_t1w[:3, :3].T + to_t1w[:3, 3])
    white, pial = surfaces
    if white.shape != pial.shape:
        raise ValueError(
            f'White and pial surfaces of sub-{sid} differ in shape: {white.shape}, {pial.shape}')
    return white, pial

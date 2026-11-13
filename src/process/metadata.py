"""Read BIDS metadata and recover parameters for distortion correction.

Resampling raw BOLD with fMRIPrep's fieldmaps requires the phase-encoding
direction and the total readout time used for susceptibility distortion
correction (SDC). fMRIPrep does not save that readout time in its outputs.
Its 25.x ``DistortionParameters`` interface computes it from inherited BIDS
metadata and the ``--fallback-total-readout-time`` option::

    get_trt(layout.get_metadata(bold_file), bold_file,
            use_estimate=(fallback == 'estimated'),
            fallback=fallback if isinstance(fallback, float) else None)

Here ``get_trt`` is ``sdcflows.utils.epimanip.get_trt``. This module provides
three entry points:

* :func:`get_bids_metadata` reads inherited JSON metadata without PyBIDS.
* :func:`get_trt` delegates to sdcflows when importable, otherwise using the
  restricted BIDS-only calculation in :func:`get_trt_bids`.
* :func:`distortion_parameters` combines metadata lookup, phase-encoding
  direction validation, and readout-time calculation for a raw BOLD file.

The BIDS-only calculation either agrees with sdcflows or raises rather than
substituting a different estimate. Readout times are in seconds; phase-encoding
directions refer to the input image's voxel axes. These helpers do not apply
spatial transforms or reorient images.
"""

import os
import json
import warnings

import nibabel as nib

# Entities defined by the BIDS specification (schema version 1.11.2).
BIDS_ENTITIES = {
    'sub', 'ses', 'sample', 'task', 'tracksys', 'acq', 'nuc', 'voi', 'ce', 'trc',
    'stain', 'rec', 'dir', 'run', 'mod', 'echo', 'flip', 'inv', 'mt', 'part',
    'proc', 'hemi', 'space', 'split', 'recording', 'chunk', 'atlas', 'seg',
    'scale', 'res', 'den', 'label', 'desc', 'tpl', 'cohort'}

# Entities whose values are indices (non-negative integers, zero padding
# optional) in the BIDS specification, so `run-1` and `run-01` are the same run.
INDEX_ENTITIES = {'run', 'echo', 'flip', 'inv', 'split', 'chunk'}

PE_DIRECTIONS = ('i', 'i-', 'j', 'j-', 'k', 'k-')


def parse_bids_filename(fn):
    """Split a BIDS filename into its entities, suffix, and extension.

    Parameters
    ----------
    fn : str or os.PathLike
        A BIDS filename or path, e.g., ``sub-01_task-rest_run-1_bold.nii.gz``.

    Returns
    -------
    entities : dict
        Entity values as strings, plus ``'suffix'`` and ``'ext'``. The suffix
        is the last underscore-separated component before the extension.
        The extension includes the first dot and everything after it, or
        is an empty string when there is no dot. Zero padding is preserved.

    Notes
    -----
    Only the basename is parsed; the file need not exist. This is a parser,
    not a BIDS filename validator: components without a hyphen are ignored
    before the suffix, and repeated entity keys keep their last value.

    Examples
    --------
    >>> parse_bids_filename('sub-01_task-rest_run-01_bold.nii.gz')
    {'sub': '01', 'task': 'rest', 'run': '01', 'suffix': 'bold', 'ext': '.nii.gz'}
    """
    stem, dot, ext = os.path.basename(fn).partition('.')
    *parts, suffix = stem.split('_')
    entities = dict(_.split('-', 1) for _ in parts if '-' in _)
    entities.update(suffix=suffix, ext=dot + ext)
    return entities


def _same_entity_value(key, a, b):
    """Compare string values numerically for digit-only index entities."""
    if key in INDEX_ENTITIES and a.isdigit() and b.isdigit():
        return int(a) == int(b)
    return a == b


def get_bids_metadata(fn, bids_root):
    """Read the metadata of a BIDS file, applying the inheritance principle.

    A JSON sidecar applies to ``fn`` if it is in the BIDS root or in a
    directory between the root and ``fn``, has the same suffix, and all of its
    entities have the same values in ``fn`` (as integers for index entities
    such as ``run``, so ``run-1`` matches ``run-01``). Applicable sidecars are
    merged from the root down to the directory of ``fn``, so values in deeper
    directories override those in higher ones.

    Parameters
    ----------
    fn : str or os.PathLike
        Path to the data file, e.g., a ``*_bold.nii.gz`` file. Relative paths
        are resolved against the current working directory, not ``bids_root``.
        The data file is not opened, but its parent directories must exist.
    bids_root : str or os.PathLike
        Dataset root at which to stop inheritance. Must contain ``fn``.

    Returns
    -------
    metadata : dict
        Merged JSON key-value pairs, or an empty dictionary if no sidecar
        applies. Merging replaces whole values rather than recursively
        merging nested objects. Top-level keys whose final value is JSON
        ``null`` are removed, including keys that had an inherited value.

    Raises
    ------
    ValueError
        If ``fn`` is outside ``bids_root``, a same-suffix JSON filename in
        a searched directory has an unknown entity, or multiple applicable
        sidecars in one directory do not form a strictly nested chain of
        entity sets. Unknown entities are rejected even if other entities
        in that filename would exclude the sidecar from matching ``fn``.
    OSError
        If a searched directory or applicable sidecar cannot be read.
    json.JSONDecodeError
        If an applicable sidecar is not valid JSON.

    Warns
    -----
    UserWarning
        If several sidecars in one directory apply and their entity sets
        form a strictly nested chain. They are merged from fewest to most
        entities, so the most specific values win. BIDS disallows multiple
        applicable sidecars at the same directory level; this is a permissive
        extension for unambiguous cases.

    Notes
    -----
    Index entities are compared numerically; labels remain case-sensitive
    strings (so ``sub-1`` and ``sub-01`` are distinct). Paths are made
    absolute without resolving symbolic links. JSON files are read as UTF-8.
    The directory search and entity checks are not a full BIDS validation.
    """
    fn = os.path.abspath(fn)
    bids_root = os.path.abspath(bids_root)
    if os.path.commonpath([fn, bids_root]) != bids_root:
        raise ValueError(f'{fn} is not inside {bids_root}')
    entities = parse_bids_filename(fn)

    dirs = [bids_root]
    rel_dir = os.path.relpath(os.path.dirname(fn), bids_root)
    if rel_dir != '.':
        for part in rel_dir.split(os.sep):
            dirs.append(os.path.join(dirs[-1], part))

    metadata = {}
    for d in dirs:
        sidecars = []
        for name in sorted(os.listdir(d)):
            ents = parse_bids_filename(name)
            ext, suffix = ents.pop('ext'), ents.pop('suffix')
            if ext != '.json' or suffix != entities['suffix']:
                continue
            unknown = ents.keys() - BIDS_ENTITIES
            if unknown:
                raise ValueError(
                    f'{os.path.join(d, name)} has entities not defined by BIDS: '
                    f'{sorted(unknown)}')
            if all(k in entities and _same_entity_value(k, entities[k], v)
                   for k, v in ents.items()):
                sidecars.append((ents.keys(), os.path.join(d, name)))

        if len(sidecars) > 1:
            sidecars.sort(key=lambda sidecar: len(sidecar[0]))
            names = [os.path.basename(path) for _, path in sidecars]
            if not all(a < b for (a, _), (b, _) in zip(sidecars, sidecars[1:])):
                raise ValueError(
                    f'Several sidecars in {d} apply to {fn}, and none of them '
                    f'is the most specific: {names}')
            warnings.warn(
                f'Several sidecars in {d} apply to {fn}, which BIDS does not '
                f'allow; the most specific one takes precedence: {names}')

        for _, json_fn in sidecars:
            with open(json_fn, encoding='utf-8') as f:
                metadata.update(json.load(f))
    return {k: v for k, v in metadata.items() if v is not None}


def get_trt_bids(in_meta, in_file=None, use_estimate=False):
    """Calculate total readout time using only standard BIDS fields.

    Uses ``TotalReadoutTime`` if present. Otherwise uses
    ``EffectiveEchoSpacing * (ReconMatrixPE - 1)``, the relation given in
    the BIDS definition of ``TotalReadoutTime``, where ``ReconMatrixPE`` is
    the number of voxels of ``in_file`` along ``PhaseEncodingDirection``.
    The matrix size is read from the image shape, not a metadata field named
    ``ReconMatrixPE``. The direction's trailing minus sign does not affect
    the readout time.

    Any other case raises ValueError rather than guessing, including when
    ``use_estimate`` is set and ``EstimatedTotalReadoutTime`` is present
    (sdcflows would prefer the estimate over ``EffectiveEchoSpacing``).

    Parameters
    ----------
    in_meta : dict
        Inherited BIDS metadata of the EPI scan. Timing values must be in
        seconds. This function does not read sidecars or validate all values.
    in_file : str or os.PathLike or None, optional
        EPI image whose shape supplies the phase-encoding matrix size.
        Required only for the effective-echo-spacing calculation.
    use_estimate : bool, optional
        Whether fMRIPrep was allowed to use dcm2niix's ``Estimated*`` fields.
        Default is False. This flag does not enable estimates here; it
        prevents returning a BIDS calculation when sdcflows would prefer
        ``EstimatedTotalReadoutTime`` instead.

    Returns
    -------
    trt : float
        Total readout time in seconds.

    Raises
    ------
    ValueError
        If ``TotalReadoutTime`` is present but false-valued (e.g., zero or
        None), or the supported calculation cannot be performed. This
        includes missing echo spacing, direction, or image path, and an
        estimate that would take precedence over echo spacing.

    Notes
    -----
    A present ``TotalReadoutTime`` is returned unchanged after the truth-value
    check, matching sdcflows; an invalid value is not replaced by an echo-spacing
    calculation. Vendor-specific fields and numerical fallback values are not
    supported. Image loading errors propagate to the caller.

    Examples
    --------
    >>> get_trt_bids({'TotalReadoutTime': 0.05})
    0.05
    """
    hint = 'Install sdcflows to handle other ways of specifying the readout time.'
    if 'TotalReadoutTime' in in_meta:
        trt = in_meta['TotalReadoutTime']
        if not trt:
            raise ValueError(f'Invalid TotalReadoutTime: {trt!r}')
        return trt

    ees = in_meta.get('EffectiveEchoSpacing')
    pe_dir = in_meta.get('PhaseEncodingDirection')
    if not ees or not pe_dir:
        raise ValueError(
            'Need TotalReadoutTime, or EffectiveEchoSpacing and '
            f'PhaseEncodingDirection. {hint}')
    if use_estimate and 'EstimatedTotalReadoutTime' in in_meta:
        raise ValueError(f'EstimatedTotalReadoutTime would take precedence. {hint}')
    if in_file is None:
        raise ValueError('`in_file` is needed to compute the total readout time')
    recon_matrix_pe = nib.load(in_file).shape['ijk'.index(pe_dir[0])]
    return ees * (recon_matrix_pe - 1)


def get_trt(in_meta, in_file=None, use_estimate=False, fallback=None):
    """Get total readout time with sdcflows or a restricted BIDS-only reader.

    Calls ``sdcflows.utils.epimanip.get_trt`` if it can be imported, which
    is what fMRIPrep uses. Otherwise falls back to :func:`get_trt_bids`, which
    covers the BIDS-defined fields and raises an error in all other cases
    (dcm2niix ``Estimated*`` fields, ``EchoSpacing`` with in-plane
    acceleration, Philips ``WaterFatShift``, ``fallback``). Either the two
    paths give the same value, or the second one raises.

    Parameters
    ----------
    in_meta : dict
        Inherited BIDS metadata of the EPI scan; timing values are in seconds.
        Sidecars are not read by this function.
    in_file : str or os.PathLike or None, optional
        EPI image used when a calculation needs the phase-encoding matrix size.
    use_estimate : bool, optional
        Allow sdcflows to use dcm2niix's ``Estimated*`` fields. Default is False.
        See :func:`get_trt_bids` for behavior when sdcflows is unavailable.
    fallback : float or None, optional
        Readout time in seconds to pass to sdcflows as a fallback. Default is
        None. Without sdcflows, this value is unused: supported BIDS fields
        still work, but a calculation requiring the fallback raises.

    Returns
    -------
    trt : float
        Total readout time in seconds.

    Raises
    ------
    ValueError
        If the selected reader cannot determine a readout time from the
        supplied inputs. Other reader and image loading errors propagate.

    Notes
    -----
    The BIDS-only reader is selected on an import failure, not on a failed
    sdcflows calculation. Errors from sdcflows do not trigger a second attempt.
    """
    try:
        from sdcflows.utils.epimanip import get_trt as sdcflows_get_trt
    except ImportError:
        return get_trt_bids(in_meta, in_file, use_estimate=use_estimate)
    return sdcflows_get_trt(in_meta, in_file, use_estimate=use_estimate, fallback=fallback)


def distortion_parameters(bold_fn, bids_root, fallback=None):
    """Read and validate the distortion parameters for a raw BOLD file.

    Mirrors ``fmriprep.interfaces.resampling.DistortionParameters``, except
    that missing information raises an error instead of being skipped, so
    that SDC is never silently dropped.

    Parameters
    ----------
    bold_fn : str or os.PathLike
        Path to the raw BOLD image whose metadata and shape were used by
        fMRIPrep. Relative paths are resolved against the working directory.
    bids_root : str or os.PathLike
        Dataset root used for JSON metadata inheritance.
    fallback : None, 'estimated', or float, optional
        Value of fMRIPrep's ``--fallback-total-readout-time`` option. None
        disables estimates and a numerical fallback; ``'estimated'`` enables
        dcm2niix estimates; a float supplies a fallback time in seconds.
        The caller must supply the option used for preprocessing: it is not
        inferred from fMRIPrep outputs. Estimates and numerical fallbacks
        require sdcflows when the standard BIDS fields do not suffice.

    Returns
    -------
    pe_direction : str
        One of ``'i'``, ``'i-'``, ``'j'``, ``'j-'``, ``'k'``, or ``'k-'``.
        These refer to the raw image's first, second, or third voxel axis;
        a trailing minus sign indicates the negative direction.
    readout_time : float
        Total readout time in seconds, without applying the phase-encoding
        direction's sign. No axis reorientation is performed.

    Raises
    ------
    ValueError
        If metadata inheritance fails, the phase-encoding direction is
        missing or invalid, or the selected reader cannot determine the
        readout time. File and JSON errors propagate to the caller.

    See Also
    --------
    get_bids_metadata : Metadata inheritance rules and ambiguity handling.
    get_trt : Readout-time calculation and optional sdcflows support.
    """
    meta = get_bids_metadata(bold_fn, bids_root)
    pe_direction = meta.get('PhaseEncodingDirection')
    if pe_direction not in PE_DIRECTIONS:
        raise ValueError(
            f'PhaseEncodingDirection of {bold_fn} is {pe_direction!r}, '
            f'expected one of {PE_DIRECTIONS}')
    readout_time = get_trt(
        meta, bold_fn,
        use_estimate=(fallback == 'estimated'),
        fallback=fallback if isinstance(fallback, float) else None,
    )
    return pe_direction, readout_time

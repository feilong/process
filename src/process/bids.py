"""BIDS dataset organization: filenames, entities, and metadata inheritance.

These helpers read a BIDS dataset with the standard library only (no PyBIDS):

* :func:`parse_bids_filename` splits a filename into entities, suffix, and
  extension.
* :func:`get_bids_metadata` merges the JSON sidecars that apply to a file,
  following the BIDS inheritance principle.
"""

import os
import json
import warnings

# Entities defined by the BIDS specification (schema version 1.11.2).
BIDS_ENTITIES = {
    'sub', 'ses', 'sample', 'task', 'tracksys', 'acq', 'nuc', 'voi', 'ce', 'trc',
    'stain', 'rec', 'dir', 'run', 'mod', 'echo', 'flip', 'inv', 'mt', 'part',
    'proc', 'hemi', 'space', 'split', 'recording', 'chunk', 'atlas', 'seg',
    'scale', 'res', 'den', 'label', 'desc', 'tpl', 'cohort'}

# Entities whose values are indices (non-negative integers, zero padding
# optional) in the BIDS specification, so `run-1` and `run-01` are the same run.
INDEX_ENTITIES = {'run', 'echo', 'flip', 'inv', 'split', 'chunk'}


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

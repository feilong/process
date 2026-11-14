"""Evaluate fMRIPrep B-spline fieldmaps as voxel shifts for raw BOLD sampling.

fMRIPrep (through sdcflows) stores each fieldmap as cubic B-spline
coefficients in Hz (``*_desc-coeff*_fieldmap.nii.gz``, one 3D file per
level), on grids defined in the fieldmap's world space, and the transform from
the BOLD reference to the fieldmap (``*_from-boldref_to-<fmapid>_mode-image*_xfm.txt``).

:func:`load_fieldmap` rewrites this B-spline in the voxel space of the raw
BOLD data and in units of voxel shifts, and :func:`fieldmap_shift` evaluates
it directly at any BOLD voxel coordinates. The fieldmap is never resampled to
a grid, so the BOLD data can be corrected with a single interpolation.

Coordinate frames
-----------------
Lookup positions are in the *BOLD reference* (boldref) frame, expressed on
the raw BOLD voxel lattice using ``bold_affine``. They are not voxel indices
of an individual moving volume, nor necessarily indices of a separately
resampled boldref image. Each raw volume has its own *moving* frame;
fMRIPrep's head-motion transforms map boldref positions to the corresponding
positions in that volume.

As in fMRIPrep, the field is evaluated before head motion is applied. For
boldref-frame coordinates ``ijk_ref`` on the raw BOLD lattice, sample volume
``t`` at ``hmc_t(ijk_ref)`` plus ``fieldmap_shift(fieldmap, ijk_ref)`` along
the raw phase-encoding axis. Here ``hmc_t`` must map between voxel coordinates
on that lattice; a world-to-world motion matrix must first be converted to
voxel coordinates. This module evaluates shifts only: it does not apply
motion transforms or interpolate BOLD intensities.

The affine of each level maps a lookup coordinate to an index of the
coefficient array, through::

    BOLD voxel -> BOLD world -> fieldmap world -> coefficient-grid index
       bold_affine     inv(xfm)      inv(coefficient affine)

where ``xfm`` is the RAS+ matrix loaded by NiTransforms from the
``from-boldref_to-<fmapid>`` transform.
That transform maps fieldmap world coordinates to boldref world coordinates,
following the ITK convention that a transform maps points of its target
image to points of its source image; its inverse is used here.

Shifts
------
The shift is in voxels of the raw BOLD data along its phase-encoding axis:
``fieldmap (Hz) * readout_time (s)`` for 'i', 'j', 'k', and its negative for
'i-', 'j-', 'k-'. It is added to the sampling coordinate along that axis;
e.g., with 'j', a +1 Hz field means that the signal of a boldref position
appears ``readout_time`` voxels toward higher j indices, and is sampled there.
This is fMRIPrep's convention (``fmriprep.interfaces.resampling.resample_vol``).
The signs refer to the raw voxel axes, as in the BIDS definition of
``PhaseEncodingDirection``; fMRIPrep reorients the data to positive direction
cosines first and flips the sign when that flips the phase-encoding axis,
which gives the same displacement.

Evaluation
----------
The represented spline is ``sum_k c_k B(x - k)`` over the coefficient-grid
indices ``k`` of each level, summed over levels, where ``B`` is the
tensor-product cubic B-spline and ``c_k`` are the stored coefficients
(no prefiltering, as in sdcflows). It is evaluated directly, up to
floating-point precision. Coefficients beyond the array are zero, so the
contribution of each level extends less than two of its own grid spacings
beyond the outermost coefficient centers along each axis. It is zero outside
that support. This boundary behavior describes the represented spline, not a
mask of valid BOLD coverage. Evaluation requires SciPy, not sdcflows.
"""

import numbers

import nibabel as nib
import nitransforms as nt
import numpy as np
from scipy.ndimage import map_coordinates

from .metadata import PE_DIRECTIONS


def load_fieldmap(coeff_fns, xfm_fn, bold_affine, pe_direction, readout_time):
    """Load coefficient levels and prepare them for voxel-shift evaluation.

    A run without SDC should not call this function; the caller handles it
    separately, so that a missing fieldmap is never mistaken for a zero one.

    Parameters
    ----------
    coeff_fns : iterable of str or os.PathLike
        The B-spline coefficient files (``*_desc-coeff*_fieldmap.nii.gz``)
        of one fieldmap, in Hz, one 3D image per level. The iterable must be
        nonempty and is consumed once. Pass a collection even for a single
        file. All levels contribute additively; their spacings may differ.
    xfm_fn : str or os.PathLike
        The ``*_from-boldref_to-<fmapid>_mode-image*_xfm.txt`` file of the run.
        Its inverse maps boldref world coordinates to fieldmap world
        coordinates.
    bold_affine : array_like of shape (4, 4)
        Voxel-to-RAS+ world affine of the raw BOLD image. This defines the
        lattice on which boldref-frame lookup coordinates are expressed.
        Use the affine and phase-encoding direction for the same image
        orientation; this function does not reorient either one.
    pe_direction : str
        Phase-encoding direction of the raw BOLD data, one of 'i', 'i-', 'j',
        'j-', 'k', 'k-'. The letters select voxel axes 0, 1, and 2; the
        trailing minus sign reverses the displacement along that voxel axis.
    readout_time : real scalar
        Total readout time in seconds, finite and strictly positive. Python
        and NumPy real numeric scalars are accepted; booleans and arrays are
        rejected. Use the readout time used by fMRIPrep for this run.

    Returns
    -------
    fieldmap : dict
        Dictionary with ``'pe_axis'``, the integer raw BOLD voxel axis along
        which to add shifts, and ``'levels'``, a list of ``(coef, affine)``
        pairs. Each ``coef`` is a 3D array of coefficients multiplied by the
        signed readout time, in voxels of shift. Each ``affine`` is a 4-by-4
        matrix mapping lookup coordinates to indices of that coefficient
        array. The result can be reused across sampling depths and volumes.

    Raises
    ------
    ValueError
        If ``coeff_fns`` is empty, ``pe_direction`` is not one of the six
        allowed directions, ``readout_time`` is not a finite positive real
        scalar, or a coefficient image is not 3D.
    numpy.linalg.LinAlgError
        If a matrix that must be inverted is singular.

    Notes
    -----
    File loading and transform parsing errors propagate from NiBabel and
    NiTransforms. The loader does not check that files belong to the intended
    run or validate all image data and affine values.

    Metadata inheritance and readout-time calculation are handled separately
    by :func:`process.metadata.distortion_parameters`. This function loads
    all coefficient arrays into memory and prepares their coordinate maps;
    it does not reconstruct a dense fieldmap.

    See Also
    --------
    fieldmap_shift : Evaluate the loaded levels at lookup coordinates.
    """
    if pe_direction not in PE_DIRECTIONS:
        raise ValueError(f'pe_direction must be one of {PE_DIRECTIONS}, got {pe_direction!r}')
    if (isinstance(readout_time, (bool, np.bool_)) or not isinstance(readout_time, numbers.Real)
            or not np.isfinite(readout_time) or readout_time <= 0):
        raise ValueError(
            f'readout_time must be a finite positive number of seconds, got {readout_time!r}')
    coeff_fns = list(coeff_fns)
    if not coeff_fns:
        raise ValueError('coeff_fns is empty; a fieldmap needs at least one coefficient file')

    pe_axis = 'ijk'.index(pe_direction[0])
    signed_readout_time = -readout_time if pe_direction.endswith('-') else readout_time
    bold_to_fmap = np.linalg.inv(nt.linear.Affine.from_filename(xfm_fn).matrix) @ bold_affine
    levels = []
    for fn in coeff_fns:
        img = nib.load(fn)
        if img.ndim != 3:
            raise ValueError(f'Coefficient image {fn} must be 3D, but has shape {img.shape}')
        coef = img.get_fdata() * signed_readout_time
        levels.append((coef, np.linalg.inv(img.affine) @ bold_to_fmap))
    return {'pe_axis': pe_axis, 'levels': levels}


def fieldmap_shift(fieldmap, ijk):
    """Evaluate signed voxel shifts at boldref-frame sampling positions.

    Parameters
    ----------
    fieldmap : dict
        Output of :func:`load_fieldmap`, containing the scaled coefficient
        levels and their coordinate maps. Reuse it for all evaluations of
        the same run; it is not modified.
    ijk : array_like of shape (3, ...)
        Boldref-frame positions in voxel coordinates of the raw BOLD data,
        i.e., before head-motion transforms are applied. Coordinates may be
        fractional and are converted to float64. The leading axis contains
        i, j, and k; trailing dimensions can represent points, depths, or a
        volume grid. A single point has shape ``(3,)``. Empty point arrays
        and non-contiguous arrays are supported. The input is not modified.

    Returns
    -------
    shift : ndarray of float64, shape (...)
        The shift (in voxels) to add to the sampling coordinates along
        ``fieldmap['pe_axis']``, so that the distorted data are sampled at the
        right place. Its shape is ``np.shape(ijk)[1:]``; a single input point
        returns a zero-dimensional array. This is a scalar displacement per
        point, not a three-component displacement vector.

    Raises
    ------
    ValueError
        If the coordinate array has no leading axis of length 3. Errors
        converting nonnumeric coordinates to float64 also propagate.

    Notes
    -----
    Each level is evaluated as a tensor-product cubic B-spline, without
    prefiltering, and the results are summed. Coefficients outside each
    array are zero; evaluation remains nonzero just outside an array when
    nearby coefficients have support there. No BOLD field-of-view clipping
    or coverage calculation is performed.

    Examples
    --------
    A constant coefficient field evaluates to one away from its boundaries:

    >>> fieldmap = {'pe_axis': 1,
    ...             'levels': [(np.ones((5, 5, 5)), np.eye(4))]}
    >>> shift = fieldmap_shift(fieldmap, [2, 2, 2])
    >>> shift.shape
    ()
    >>> bool(np.isclose(shift, 1.0))
    True

    See Also
    --------
    load_fieldmap : Load and scale fMRIPrep coefficient files for a BOLD run.
    """
    ijk = np.asarray(ijk, dtype=np.float64)
    if ijk.ndim < 1 or ijk.shape[0] != 3:
        raise ValueError(f'ijk must have shape (3, ...), got {ijk.shape}')
    points = ijk.reshape(3, -1)
    shift = np.zeros(points.shape[1])
    for coef, affine in fieldmap['levels']:
        grid = affine[:3, :3] @ points + affine[:3, 3:]
        shift += map_coordinates(
            coef, grid, order=3, prefilter=False, mode='grid-constant', cval=0.0)
    return shift.reshape(ijk.shape[1:])

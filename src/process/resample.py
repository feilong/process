import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates, spline_filter
import warnings
import h5py
from nitransforms.io.itk import ITKCompositeH5


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

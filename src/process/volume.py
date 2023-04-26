import os
import numpy as np
import nibabel as nib

__all__ = ['canonical_volume_coords', 'mni_affine', 'mni_coords',
    'aseg_mapping', 'extract_data_in_mni', 'resample_mni_to_resolution']

DIR = os.path.dirname(os.path.abspath(__file__))


def find_truncation_boundaries(brainmask, margin=2):
    boundaries = np.zeros((3, 2), dtype=int)
    for dim in range(3):
        mask = np.all(brainmask == 0, axis=tuple(_ for _ in range(3) if _ != dim))
        for i in range(brainmask.shape[dim]):
            if mask[i]:
                boundaries[dim, 0] = i
            else:
                break
        for i in range(brainmask.shape[dim])[::-1]:
            if mask[i]:
                boundaries[dim, 1] = i
            else:
                break
    boundaries[:, 0] -= margin
    boundaries[:, 1] += margin
    boundaries[:, 0] = np.maximum(boundaries[:, 0], 0)
    boundaries[:, 1] = np.minimum(boundaries[:, 1] + 1, brainmask.shape)
    return boundaries


def canonical_volume_coords(brainmask, margin=2):
    canonical = nib.as_closest_canonical(brainmask)
    boundaries = find_truncation_boundaries(np.asarray(canonical.dataobj), margin=margin)
    coords = np.mgrid[boundaries[0, 0]:boundaries[0, 1], boundaries[1, 0]:boundaries[1, 1], boundaries[2, 0]:boundaries[2, 1], 1:2].astype(np.float64)[..., 0]
    coords = np.moveaxis(coords, 0, -1) @ canonical.affine.T
    return coords


mni_affine = np.array(
    [[   1.,    0.,    0.,  -96.],
     [   0.,    1.,    0., -132.],
     [   0.,    0.,    1.,  -78.],
     [   0.,    0.,    0.,    1.]])


def _get_MNI_coords():
    ijk1 = [
        np.tile(np.arange(193)[:, np.newaxis, np.newaxis, np.newaxis], (1, 229, 193, 1)),
        np.tile(np.arange(229)[np.newaxis, :, np.newaxis, np.newaxis], (193, 1, 193, 1)),
        np.tile(np.arange(193)[np.newaxis, np.newaxis, :, np.newaxis], (193, 229, 1, 1)),
        np.ones((193, 229, 193, 1)),
    ]
    ijk1 = np.concatenate(ijk1, axis=3)
    xyz1 = ijk1 @ mni_affine.T
    return xyz1

mni_coords = _get_MNI_coords()


def resample_mni_to_resolution(data, mm=1):
    assert data.shape[:3] == (193, 229, 193)
    extra_dims = data.shape[3:]
    affine = np.eye(4)
    affine[np.arange(3), np.arange(3)] = mm
    affine[:3, 3] = np.array([-96, -132, -78]) - (mm - 1) * 0.5

    target_shapes = {
        1: [193, 229, 193],
        2: [97, 115, 97],
        3: [65, 77, 65],
        4: [49, 58, 49],
    }
    target_shape = target_shapes[mm]
    reshape = [_ * mm for _ in target_shape]

    for i in range(3):
        s_ = list(data.shape)
        s_[i] = mm - 1
        pad = np.zeros(s_, dtype=data.dtype)
        data = np.concatenate([pad, data], axis=i)
    data = data.reshape(
        target_shape[0], mm, target_shape[1], mm, target_shape[2], mm, *extra_dims)
    data = data.mean(axis=(1, 3, 5))
    assert list(data.shape)[:3] == target_shape
    return data, affine


aseg_mapping = {
    3:  'l-cerebrum',
    42: 'r-cerebrum',

    8:  'l-cerebellum',
    47: 'r-cerebellum',
    16: 'brain-stem',

    10: 'l-thalamus',
    49: 'r-thalamus',
    11: 'l-caudate',
    50: 'r-caudate',
    12: 'l-putamen',
    51: 'r-putamen',
    13: 'l-pallidum',
    52: 'r-pallidum',
    17: 'l-hippocampus',
    53: 'r-hippocampus',
    18: 'l-amygdala',
    54: 'r-amygdala',
    26: 'l-accumbens',
    58: 'r-accumbens',
    28: 'l-ventral-diencephalon',
    60: 'r-ventral-diencephalon',
}


def extract_data_in_mni(data, mm=2, cortex=True):
    resampled, affine = resample_mni_to_resolution(data, mm=mm)
    atlas = nib.load(os.path.join(DIR, 'data',
        f'tpl-MNI152NLin2009cAsym_res-{mm:02d}_atlas-aseg_dseg.nii.gz'))
    atlas = np.asanyarray(atlas.dataobj)
    output = {}
    for idx, name in aseg_mapping.items():
        if idx in [3, 42] and not cortex:
            continue
        mask = (atlas == idx)
        roi = resampled[mask]
        output[name] = roi
    return output

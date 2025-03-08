import os
import scipy.sparse as sparse
import neuroboros as nb

from .surface import Hemisphere


def xform_workflow(sid, fs_dir, xform_dir, combinations=None, pairs=None):
    assert combinations is not None or pairs is not None
    if pairs is None:
        pairs = set()
        for a, b in combinations[::-1]:
            b, c = b.split('_', 1)
            c, d = c.rsplit('_', 1)
            pairs.add((a, d)) # (space, resample)

    for lr in 'lr':
        hemi = Hemisphere(lr, fs_dir)

        for space, resample in pairs:
            if space == 'native':
                continue
            sphere_coords = nb.geometry('sphere.reg', lr, space, vertices_only=True)

            xform_fn = os.path.join(xform_dir, space, f'{sid}_{resample}_{lr}h.npz')
            if not os.path.exists(xform_fn):
                os.makedirs(os.path.dirname(xform_fn), exist_ok=True)
                xform = hemi.get_transformation(sphere_coords, space, resample)
                sparse.save_npz(xform_fn, xform)

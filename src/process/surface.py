from glob import glob
import numpy as np
import scipy.sparse as sparse
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates, spline_filter
import nibabel as nib
import nitransforms as nt


def compute_vertex_normals_sine_weight(coords, faces):
    normals = np.zeros(coords.shape)

    f_coords = coords[faces]
    edges = np.roll(f_coords, 1, axis=1) - f_coords
    del f_coords
    edges /= np.linalg.norm(edges, axis=2, keepdims=True)

    for f, ee in zip(faces, edges):
        normals[f[0]] += np.cross(ee[0], ee[1])
        normals[f[1]] += np.cross(ee[1], ee[2])
        normals[f[2]] += np.cross(ee[2], ee[0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    return normals


def compute_vertex_normals_equal_weight(coords, faces):
    normals = np.zeros(coords.shape)

    f_coords = coords[faces]
    e01 = f_coords[:, 1, :] - f_coords[:, 0, :]
    e12 = f_coords[:, 2, :] - f_coords[:, 1, :]
    del f_coords

    face_normals = np.cross(e01, e12)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)
    for f, n in zip(faces, face_normals):
        normals[f] += n
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    return normals


# def compute_projected_coordinates(white_coords, normals, thicknesses, frac):
#     return white_coords + normals * (thicknesses[:, np.newaxis] * frac)


def nnfr_transformation(source_sphere, target_sphere, reverse=True):
    ns = source_sphere.shape[0]
    nt = target_sphere.shape[0]
    source_tree = cKDTree(source_sphere)
    target_tree = cKDTree(target_sphere)

    forward_indices = source_tree.query(target_sphere)[1]
    if reverse:
        u, c = np.unique(forward_indices, return_counts=True)
        counts = np.zeros((ns, ), dtype=int)
        counts[u] += c
        remaining = np.setdiff1d(np.arange(ns), u)
        reverse_indices = target_tree.query(source_sphere[remaining])[1]
        counts[remaining] += 1

    T = sparse.lil_matrix((ns, nt))
    for t_idx, s_idx in zip(np.arange(nt), forward_indices):
        T[s_idx, t_idx] += 1
    if reverse:
        for t_idx, s_idx in zip(reverse_indices, remaining):
            T[s_idx, t_idx] += 1

    T = T.tocsr()
    t_counts = T.sum(axis=0).A.ravel()
    T = T @ sparse.diags(np.reciprocal(t_counts))

    return T


# def compute_transformed_ijk(coords, c_ras, xfms):
#     coords = coords + c_ras[np.newaxis]
#     coords = np.concatenate([coords, np.ones(coords.shape[:-1] + (1, ), dtype=coords.dtype)], axis=1)
#     for xfm in xfms:
#         coords = coords @ xfm.T
#     return coords


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
    boundaries = find_truncation_boundaries(np.asarray(canonical.dataobj))
    coords = np.mgrid[boundaries[0, 0]:boundaries[0, 1], boundaries[1, 0]:boundaries[1, 1], boundaries[2, 0]:boundaries[2, 1], 1:2].astype(np.float64)[..., 0]
    shape = coords.shape[1:]
    coords = coords.reshape(4, -1).T @ canonical.affine.T
    return coords, shape


def surface_coords_normal(white_coords, c_ras, normals, thicknesses, fracs=np.linspace(0, 1, 6)):
    coords = (white_coords[:, np.newaxis, :] + 
              c_ras[np.newaxis, np.newaxis, :] +
              normals[:, np.newaxis, :] * thicknesses[:, np.newaxis, np.newaxis] * fracs[np.newaxis, :, np.newaxis])
    shape = coords.shape[:-1]
    coords = coords.reshape(-1, 3)
    coords = np.concatenate([coords, np.ones(coords.shape[:-1] + (1, ), dtype=coords.dtype)], axis=-1)
    return coords, shape


def surface_coords_pial(white_coords, c_ras, pial_coords, fracs=np.linspace(0, 1, 6)):
    white_coords = white_coords + c_ras[np.newaxis]
    pial_coords = pial_coords + c_ras[np.newaxis]
    fracs = fracs[np.newaxis, :, np.newaxis]
    coords = white_coords[:, np.newaxis, :] * (1 - fracs) + pial_coords[:, np.newaxis, :] * fracs
    shape = coords.shape[:-1]
    coords = coords.reshape(-1, 3)
    coords = np.concatenate([coords, np.ones(coords.shape[:-1] + (1, ), dtype=coords.dtype)], axis=-1)
    return coords, shape


def interpolate_original_space(nii_data, nii_affines, coords, shape, lta, ref_to_t1, hmc, warp_data=None, warp_affines=None, interp_kwargs={'order': 3, 'prefilter': False, 'cval': np.nan}):
    coords = coords @ (lta.T @ ref_to_t1.T)
    interps = []
    for i, (data, affine) in enumerate(zip(nii_data, nii_affines)):
        cc = coords.copy()
        if warp_data is not None:
            ijk = cc @ np.linalg.inv(warp_affines[i]).T
            for j in range(3):
                diff = map_coordinates(warp_data[i][:, :, :, 0, j], ijk.T[:3], order=1)
                cc[:, j] -= diff
        cc = cc @ hmc[i].T
        cc = cc @ np.linalg.inv(affine).T

        interp = map_coordinates(data, cc.T[:3], **interp_kwargs).reshape(shape)
        if len(interp.shape) == 2:
            interp = np.nanmean(interp, axis=1)
        interps.append(interp)
    interps = np.stack(interps, axis=0)

    return interps


class Interpolator(object):
    def __init__(self, sid, label, fs_dir, wf_dir):
        self.sid = sid
        self.label = label
        self.fs_dir = fs_dir
        self.wf_dir = wf_dir
        self.surface_data = {'l': {}, 'r': {}}

    def prepare(self):
        # T1 = nib.load(f'{fs_dir}/sub-{sid}/mri/T1.mgz')
        # c_ras = (np.array([_//2 for _ in T1.shape] + [1]) @ T1.affine.T)[:3]

        self.brainmask = nib.load(f'{self.fs_dir}/sub-{self.sid}/mri/brainmask.mgz')
        self.c_ras = (np.array([_//2 for _ in self.brainmask.shape] + [1]) @ self.brainmask.affine.T)[:3]
        # self.canonical = nib.as_closest_canonical(self.brainmask)
        self.vol_coords, self.vol_shape = canonical_volume_coords(self.brainmask, margin=2)

        for lr in 'lr':
            hemi = self.surface_data[lr]
            hemi['white'], hemi['faces'] = nib.freesurfer.io.read_geometry(f'{self.fs_dir}/sub-{self.sid}/surf/{lr}h.white')
            hemi['pial'] = nib.freesurfer.io.read_geometry(f'{self.fs_dir}/sub-{self.sid}/surf/{lr}h.pial')[0]
            hemi['thicknesses'] = nib.freesurfer.io.read_morph_data(f'{self.fs_dir}/sub-{self.sid}/surf/{lr}h.thickness')
            hemi['normals_sine'] = compute_vertex_normals_sine_weight(hemi['white'], hemi['faces'])
            hemi['normals_equal'] = compute_vertex_normals_equal_weight(hemi['white'], hemi['faces'])

            for key in ['normals_sine', 'normals_equal']:
                hemi['coords_' + key], hemi['shape'] = surface_coords_normal(
                    hemi['white'], self.c_ras, hemi[key], hemi['thicknesses'])
            hemi['coords_pial'], shape = surface_coords_pial(
                hemi['white'], self.c_ras, hemi['pial'])
            assert hemi['shape'] == shape

            hemi['native_sphere'] = nib.freesurfer.io.read_geometry(f'{self.fs_dir}/sub-{self.sid}/surf/{lr}h.sphere.reg')[0]
            hemi['fsavg_sphere'] = nib.freesurfer.io.read_geometry(f'{self.fs_dir}/fsaverage/surf/{lr}h.sphere.reg')[0]
            hemi['sphere_T'] = nnfr_transformation(hemi['native_sphere'], hemi['fsavg_sphere'])

        self.hmc = nt.io.itk.ITKLinearTransformArray.from_filename(
            f'{self.wf_dir}/bold_hmc_wf/fsl2itk/mat2itk.txt').to_ras()
        self.ref_to_t1 = nt.io.itk.ITKLinearTransform.from_filename(
            f'{self.wf_dir}/bold_reg_wf/bbreg_wf/concat_xfm/out_fwd.tfm').to_ras()
        self.lta = nt.io.lta.FSLinearTransform.from_filename(
            f'{self.wf_dir}/bold_surf_wf/itk2lta/out.lta').to_ras()

        nii_fns = sorted(glob(f'{self.wf_dir}/bold_split/vol*.nii.gz'))
        warp_fns = sorted(glob(f'{self.wf_dir}/unwarp_wf/resample/vol*_xfm.nii.gz'))
        assert len(nii_fns) == len(warp_fns)

        self.nii_data, self.nii_affines = [], []
        for i, nii_fn in enumerate(nii_fns):
            nii = nib.load(nii_fn)
            data = np.asarray(nii.dataobj)
            data = spline_filter(data, order=3)
            self.nii_data.append(data)
            self.nii_affines.append(nii.affine)

        self.warp_data, self.warp_affines = [], []
        for i, warp_fn in enumerate(warp_fns):
            warp_nii = nib.load(warp_fn)
            self.warp_data.append(np.asarray(warp_nii.dataobj))
            self.warp_affines.append(warp_nii.affine)

    def interpolate_surface(self, lr, coords_type='normals_sine', standard_space=True):
        hemi = self.surface_data[lr]
        interp = interpolate_original_space(
            self.nii_data, self.nii_affines, hemi['coords_' + coords_type], hemi['shape'],
            self.lta, self.ref_to_t1, self.hmc, self.warp_data, self.warp_affines)
        if standard_space:
            interp = interp @ hemi['sphere_T']
        return interp

    def interpolate_volume(self):
        interp = interpolate_original_space(
            self.nii_data, self.nii_affines, self.vol_coords, self.vol_shape,
            self.lta, self.ref_to_t1, self.hmc, self.warp_data, self.warp_affines)

    # 2 hemispheres x 2 styles + 1 volume

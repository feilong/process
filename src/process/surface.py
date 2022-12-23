import os
from glob import glob
from datetime import datetime
import numpy as np
import scipy.sparse as sparse
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates, spline_filter
import nibabel as nib
import nitransforms as nt
from joblib import Parallel, delayed

from surface import Surface, barycentric_resample
from surface.mapping import compute_transformation

from .resample import parse_warp_image


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


def vertex_area_transformation(source_sphere, source_faces, target_sphere, source_mid):
    T = compute_transformation(
        source_sphere, source_faces, target_sphere, source_mid)
    T = sparse.diags(np.reciprocal(T.sum(axis=1).A.ravel())) @ T
    return T


def surface_coords_normal(white_coords, c_ras, normals, thicknesses, fracs=np.linspace(0, 1, 6)):
    coords = (white_coords[:, np.newaxis, :] + 
              c_ras[np.newaxis, np.newaxis, :] +
              normals[:, np.newaxis, :] * thicknesses[:, np.newaxis, np.newaxis] * fracs[np.newaxis, :, np.newaxis])
    # shape = coords.shape[:-1]
    # coords = coords.reshape(-1, 3)
    coords = np.concatenate([coords, np.ones(coords.shape[:-1] + (1, ), dtype=coords.dtype)], axis=-1)
    # return coords, shape
    return coords


def surface_coords_pial(white_coords, c_ras, pial_coords, fracs=np.linspace(0, 1, 6)):
    white_coords = white_coords + c_ras[np.newaxis]
    pial_coords = pial_coords + c_ras[np.newaxis]
    fracs = fracs[np.newaxis, :, np.newaxis]
    coords = white_coords[:, np.newaxis, :] * (1 - fracs) + pial_coords[:, np.newaxis, :] * fracs
    # shape = coords.shape[:-1]
    # coords = coords.reshape(-1, 3)
    coords = np.concatenate([coords, np.ones(coords.shape[:-1] + (1, ), dtype=coords.dtype)], axis=-1)
    # return coords, shape
    return coords


class Hemisphere(object):
    def __init__(self, lr, fs_dir):
        self.lr = lr
        self.fs_dir = fs_dir

        self.load_data()

    def load_data(self):
        native = {}
        for key in ['white', 'pial', 'sphere.reg']:
            coords, faces = nib.freesurfer.io.read_geometry(
                os.path.join(self.fs_dir, 'surf', f'{self.lr}h.{key}'))
            native[key] = coords.astype(np.float64)
        native['faces'] = faces
        native['midthickness'] = (native['white'] + native['pial']) * 0.5

        native['thickness'] = nib.freesurfer.io.read_morph_data(
            os.path.join(self.fs_dir, 'surf', f'{self.lr}h.thickness')
        ).astype(np.float64)

        self.native = native
        # The resample method and other spaces are temporarily removed
        # because they're not used in the current version.

        T1 = nib.load(os.path.join(self.fs_dir, 'mri', 'T1.mgz'))
        self.c_ras = (np.array([_ / 2 for _ in T1.shape] + [1]) @ T1.affine.T)[:3]

    def get_coordinates(self, kind):
        assert kind in ['normals-equal', 'normals-sine', 'pial']
        space = self.native
        if f'coords_{kind}' in space:
            return space[f'coords_{kind}']

        if kind.startswith('normals-'):
            if kind not in space:
                func = {
                    'normals-equal': compute_vertex_normals_equal_weight,
                    'normals-sine': compute_vertex_normals_sine_weight,
                }[kind]
                space[kind] = func(space['white'], space['faces'])
            space[f'coords_{kind}'] = surface_coords_normal(
                space['white'], self.c_ras, space[kind], space['thickness'])
        elif kind == 'pial':
            space[f'coords_{kind}'] = surface_coords_pial(
                space['white'], self.c_ras, space['pial'])
        else:
            raise ValueError

        # if 'shape' in space:
        #     assert shape == space['shape']
        # else:
        #     space['shape'] = shape

        return space[f'coords_{kind}']

    def get_transformation(self, sphere_fn, name, method):
        key = f'to_{name}_{method}'
        if key in self.native:
            return self.native[key]

        if not hasattr(self, f'{name}_sphere'):
            setattr(self, f'{name}_sphere', np.load(sphere_fn)['coords'])
        sphere = self.native['sphere.reg'] / np.linalg.norm(self.native['sphere.reg'], axis=1, keepdims=True)
        if method == 'nnfr':
            self.native[key] = nnfr_transformation(
                sphere, getattr(self, f'{name}_sphere'))
        elif method == 'area':
            self.native[key] = vertex_area_transformation(
                sphere, self.native['faces'], getattr(self, f'{name}_sphere'), self.native['midthickness'])
        return self.native[key]

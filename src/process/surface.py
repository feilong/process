import os
import numpy as np
import scipy.sparse as sparse
import nibabel as nib
import neuroboros as nb


from neuroboros.surface import Surface, Sphere
from neuroboros.surface.voronoi import compute_overlap, subdivision_voronoi, native_voronoi, subdivide_edges, inverse_face_mapping
from neuroboros.surface.properties import compute_vertex_normals_sine_weight, compute_vertex_normals_equal_weight
from neuroboros.surface.nnfr import nnfr
from neuroboros.surface.areal import areal


def surface_coords_normal(white_coords, c_ras, normals, thicknesses, fracs=np.linspace(0, 1, 6)):
    coords = (white_coords[:, np.newaxis, :] + 
              c_ras[np.newaxis, np.newaxis, :] +
              normals[:, np.newaxis, :] * thicknesses[:, np.newaxis, np.newaxis] * fracs[np.newaxis, :, np.newaxis])
    coords = np.concatenate([coords, np.ones(coords.shape[:-1] + (1, ), dtype=coords.dtype)], axis=-1)
    return coords


def surface_coords_pial(white_coords, c_ras, pial_coords, fracs=np.linspace(0, 1, 6)):
    white_coords = white_coords + c_ras[np.newaxis]
    pial_coords = pial_coords + c_ras[np.newaxis]
    fracs = fracs[np.newaxis, :, np.newaxis]
    coords = white_coords[:, np.newaxis, :] * (1 - fracs) + pial_coords[:, np.newaxis, :] * fracs
    coords = np.concatenate([coords, np.ones(coords.shape[:-1] + (1, ), dtype=coords.dtype)], axis=-1)
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

        return space[f'coords_{kind}']

    def prepare_overlap_transformation(self, n_div=8):
        if 'nn' in self.native:
            return
        mid = Surface(self.native['midthickness'], self.native['faces'])
        self.native['face_areas'] = mid.face_areas
        new_coords, self.native['e_mapping'], self.native['neighbors'] = subdivide_edges(
            mid.coords, mid.faces, n_div=n_div)
        self.native['coords'] = np.concatenate([mid.coords, new_coords], axis=0)
        self.native['nn'], self.native['nnd'] = native_voronoi(
            self.native['coords'], mid.faces, self.native['e_mapping'], self.native['neighbors'])

    def compute_overlap_transformation(self, tmpl_coords):
        # tmpl_coords = np.load(sphere_fn)['coords']
        sphere = Sphere(self.native['sphere.reg'], self.native['faces'])
        f_indices, weights = sphere.barycentric(tmpl_coords, return_sparse=False, eps=1e-7)
        nn, nnd = subdivision_voronoi(
            self.native['coords'], self.native['faces'],
            self.native['e_mapping'], self.native['neighbors'],
            f_indices, weights)
        f_inv = inverse_face_mapping(
            f_indices, weights, self.native['coords'], self.native['faces'])
        T = compute_overlap(
            self.native['faces'], self.native['face_areas'],
            self.native['e_mapping'], self.native['coords'],
            self.native['nn'], self.native['nnd'], {},
            nn, nnd, f_inv, self.native['midthickness'].shape[0], tmpl_coords.shape[0],
        )
        # T = sparse.diags(np.reciprocal(T.sum(axis=1).A.ravel())) @ T
        return T

    def get_transformation(self, coords, name, method):
        key = f'to_{name}_{method}'
        if key in self.native:
            return self.native[key]
        if isinstance(coords, str) and coords.endswith('.npz'):
            coords = np.load(coords)['coords']

        if not hasattr(self, f'{name}_sphere'):
            setattr(self, f'{name}_sphere', coords)
        sphere = self.native['sphere.reg'] / np.linalg.norm(self.native['sphere.reg'], axis=1, keepdims=True)
        if method == 'nnfr':
            self.native[key] = nnfr(sphere, getattr(self, f'{name}_sphere'))
        elif method == 'area':
            self.native[key] = areal(
                Sphere(sphere, self.native['faces']),
                getattr(self, f'{name}_sphere'),
                self.native['midthickness'])
        elif method == 'overlap':
            self.prepare_overlap_transformation()
            self.native[key] = self.compute_overlap_transformation(getattr(self, f'{name}_sphere'))
        return self.native[key]

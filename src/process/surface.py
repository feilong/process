import numpy as np
import scipy.sparse as sparse
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
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

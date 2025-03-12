import os
import subprocess
from glob import glob
import numpy as np
import scipy.sparse as sparse
import nibabel as nib
import neuroboros as nb

"""
export FREESURFER_HOME=$HOME/lab/freesurfer
. $FREESURFER_HOME/FreeSurferEnv.sh
"""


def run_freesurfer_invivo_v1(config, log_dir):
    sid = config['sid']
    fs_dir = os.path.join(config['fmriprep_out'], 'freesurfer')
    v1_dir = os.path.join(fs_dir, 'V1_average')
    if not os.path.exists(v1_dir):
        os.symlink(
            os.path.realpath(os.path.expanduser('/dartfs/rc/lab/H/Haxbylab/shared/freesurfer/subjects/V1_average')),
            v1_dir)
    cmd = ['recon-all', '-label_v1', '-s', f'sub-{sid}', '-sd', fs_dir]
    stdout_fn = os.path.join(log_dir, f'{sid}_V1_stdout.txt')
    stderr_fn = os.path.join(log_dir, f'{sid}_V1_stderr.txt')
    with open(stdout_fn, 'w') as f1, open(stderr_fn, 'w') as f2:
        proc = subprocess.run(cmd, stdout=f1, stderr=f2)
    return proc


def resample_freesurfer(config, anat_dir, xform_dir):
    measures = [
        'area', 'area.mid', 'area.pial', 'thickness', 'volume',
        'curv', 'curv.pial', 'sulc', 'jacobian_white']
    atlases = [
        'aparc.a2009s.annot', 'aparc.annot', 'aparc.DKTatlas.annot',
    ]  # https://surfer.nmr.mgh.harvard.edu/fswiki/SurfaceLabelAtlas
    probs = ['v1.prob.label']

    # pairs = [(_[0], _[1].split('_', 2)[2]) for _ in config['combinations']]
    pairs = [(_[0], 'overlap') for _ in config['combinations']]
    pairs = sorted(set(pairs))

    sid = config['sid']

    xforms = {}
    for lr in 'lr':
        for space, resample in pairs:
            if space == 'native':
                continue
            xform_fn = os.path.join(xform_dir, space, f'{sid}_{resample}_{lr}h.npz')
            mat = sparse.load_npz(xform_fn)
            if resample.startswith('areal') or resample.startswith('overlap'):
                s = mat.sum(axis=0).A.ravel()
                diag = sparse.diags(np.reciprocal(s))
                mat1 = mat @ diag
            else:
                mat1 = mat.copy()
            s = mat.sum(axis=1).A.ravel()
            diag = sparse.diags(np.reciprocal(s))
            mat2 = diag @ mat
            xforms[lr, space, resample] = [mat1, mat2]

    for lr in 'lr':
        for measure in measures:
            fn = os.path.join(config['fmriprep_out'], 'freesurfer',
                f'sub-{sid}', 'surf', f'{lr}h.{measure}')
            d = nib.freesurfer.io.read_morph_data(fn).astype(np.float64)
            mat_idx = 1 if measure in ['area', 'area.mid', 'area.pial', 'volume'] else 0
            for space, resample in pairs:
                out_fn = os.path.join(anat_dir, space, resample, measure, f'{sid}_{lr}h.npy')
                if space != 'native':
                    mat = xforms[lr, space, resample][mat_idx]
                    nb.save(out_fn, d @ mat)
                else:
                    nb.save(out_fn, d)

        for prob in probs:
            fn = os.path.join(config['fmriprep_out'], 'freesurfer',
                f'sub-{sid}', 'label', f'{lr}h.{prob}')
            with open(fn, 'r') as f:
                lines = f.read().splitlines()
            n = int(lines[1])
            assert len(lines) == n + 2
            d = np.zeros((mat.shape[0], ))
            for line in lines[2:]:
                pp = line.split()
                d[int(pp[0])] = float(pp[4])
            for space, resample in pairs:
                out_fn = os.path.join(anat_dir, space, resample, prob, f'{sid}_{lr}h.npy')
                if space != 'native':
                    mat = xforms[lr, space, resample][0]
                    nb.save(out_fn, d @ mat)
                else:
                    nb.save(out_fn, d)

        for atlas in atlases:
            fn = os.path.join(config['fmriprep_out'], 'freesurfer',
                f'sub-{sid}', 'label', f'{lr}h.{atlas}')
            labels = nib.freesurfer.io.read_annot(fn)[0]
            uu = np.unique(labels)
            for space, resample in pairs:
                out_fn = os.path.join(anat_dir, space, resample, atlas, f'{sid}_{lr}h.npy')
                if space != 'native':
                    mat = xforms[lr, space, resample][0]
                    maps = []
                    for u in uu:
                        m = (labels == u).astype(np.float64) @ mat
                        maps.append(m)
                    indices = np.argmax(np.array(maps), axis=0)
                    d = nb.utils.optimize_dtype(uu[indices])
                    nb.save(out_fn, d)
                else:
                    d = nb.utils.optimize_dtype(labels)
                    nb.save(out_fn, d)

import os
import shutil
from glob import glob
import numpy as np
import pandas as pd

from .regression import read_nuisance_regressors, legendre_regressors


def copy_confound_files(fmriprep_out, confounds_dir):
    in_fns = sorted(glob(os.path.join(fmriprep_out, 'func', '*.tsv'))) + \
        sorted(glob(os.path.join(fmriprep_out, 'ses-*', 'func', '*.tsv')))
    for in_fn in in_fns:
        out_fn = os.path.join(confounds_dir, os.path.basename(in_fn))
        shutil.copy2(in_fn, out_fn)


def compute_regressors(confounds_dir):
    fns = sorted(glob(os.path.join(confounds_dir, '*.tsv')))
    for fn in fns:
        out_fn = fn[:-4] + '.npy'
        regressors = read_nuisance_regressors(fn, ignore_non_existing=True)
        regressors = np.hstack([
            regressors,
            legendre_regressors(polyord=2, n_tp=regressors.shape[0])])
        np.save(out_fn, regressors)


def compute_temporal_mask(confounds_dir):
    fns = sorted(glob(os.path.join(confounds_dir, '*.tsv')))
    for fn in fns:
        assert '_desc-confounds_' in fn
        out_fn = fn.replace('_desc-confounds_', '_desc-mask_')[:-4] + '.npy'
        df = pd.read_csv(fn, delimiter='\t', na_values='n/a')
        mask = np.zeros((df.shape[0], ), dtype=bool)
        for key in df:
            if key.startswith('motion_outlier'):
                mask = np.logical_or(mask, df[key])
        mask = np.logical_not(mask)
        np.save(out_fn, mask)


def confound_workflow(fmriprep_out, confounds_dir):
    copy_confound_files(fmriprep_out, confounds_dir)
    compute_regressors(confounds_dir)
    compute_temporal_mask(confounds_dir)

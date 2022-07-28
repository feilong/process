import os
from io import StringIO
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.special import legendre
import tarfile
from glob import glob
from joblib import Parallel, delayed


default_columns = \
    ['a_comp_cor_%02d' % (_, ) for _ in range(6)] + \
    ['framewise_displacement'] + \
    ['trans_x', 'trans_y', 'trans_z'] + \
    ['rot_x', 'rot_y', 'rot_z'] + \
    ['trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1'] + \
    ['rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1']


def legendre_regressors(polyord, timepoints=None, n_tp=None):
    if timepoints is None:
        timepoints = np.linspace(-1, 1, n_tp)
    else:
        timepoints -= timepoints.min()
        timepoints = (timepoints / timepoints.max()) * 2 - 1
    regressors = np.array([legendre(i)(timepoints) for i in range(polyord + 1)]).T
    return regressors


def read_nuisance_regressors(conf_fn, columns=None, ignore_non_existing=False):
    if columns is None:
        columns = default_columns
    regressors = pd.read_csv(conf_fn, delimiter='\t', na_values='n/a')
    for column in columns:
        if column not in regressors.keys():
            print(column, regressors.keys())
            if not ignore_non_existing:
                raise KeyError(f"Column {column} not found in tsv file.")

    regressors = [np.array(regressors[column]) for column in columns if column in regressors.keys()]
    regressors = np.nan_to_num(np.stack(regressors, axis=1))

    return regressors


def regression_workflow_single_run(fn, out_fn, conf_buffer, ignore_non_existing=False):
    ds = np.load(fn)
    regressors = np.hstack([
        read_nuisance_regressors(conf_buffer, ignore_non_existing=ignore_non_existing),
        legendre_regressors(polyord=2, n_tp=ds.shape[0])])
    betas = np.linalg.lstsq(regressors, ds, rcond=None)[0]
    ds = ds - regressors @ betas + ds.mean(axis=0, keepdims=True)
    np.save(out_fn, ds)


def regression_workflow(config, out_dir, rename_func, n_jobs=1, resample_flavor=None, ignore_non_existing=False):
    if resample_flavor is None:
        resample_flavor = config['resample_flavor']
    sid = config['sid']
    resample_dir = os.path.join(config['output_root'], 'resampled', resample_flavor)
    fns = sorted(glob(os.path.join(resample_dir, f'sub-{sid}*.npy')))
    # print(sid, len(fns))
    os.makedirs(out_dir, exist_ok=True)

    buffers = {}
    out_fns = {}
    confounds_fn = os.path.join(config['output_root'], 'confounds', f'{sid}.tar.lzma')
    with tarfile.open(confounds_fn, 'r') as tf:
        members = tf.getmembers()
        for fn in fns:
            out_fn = os.path.join(out_dir, rename_func(fn) + '.npy')
            if os.path.exists(out_fn):
                continue
            label = os.path.basename(fn)[:-7]
            parts = label.split('_run-')
            if parts[1].startswith('0'):
                label = '_run-'.join([parts[0], parts[1][1:]])
            mm = [_ for _ in members if label in _.name and _.name.endswith('.tsv')]
            if len(mm) != 1:
                print(mm)
                print(label)
                print(members)
            content = tf.extractfile(mm[0]).read()
            buffers[fn] = StringIO(content.decode('ascii'))
            out_fns[fn] = out_fn

    jobs = []
    for fn, buffer in buffers.items():
        job = delayed(regression_workflow_single_run)(fn, out_fns[fn], buffer, ignore_non_existing=ignore_non_existing)
        jobs.append(job)

    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(jobs)

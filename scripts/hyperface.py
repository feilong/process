import os
import sys
import json
from glob import glob
from process.fmriprep import run_fmriprep
from process.regression import regression_workflow


def rename_func(fn):
    timing_dir = os.path.expanduser('~/hyperface/timing')
    hyperface_runs = [2, 3, 4, 6, 7, 8]

    info = {}
    chunks = os.path.basename(fn).split('_')
    for chunk in chunks[:-1]:
        key, val = chunk.split('-')
        info[key] = val
    info['lr'] = chunks[-1][0]

    for key in info:
        if key not in ['sub', 'ses', 'task', 'run', 'lr']:
            raise ValueError
    sid, ses, task, run, lr = info['sub'], info['ses'], info['task'], info['run'], info['lr']

    if task == 'localizer':
        key = (ses, run)
        new_run = {('1', '01'): 1, ('1', '05'): 2, ('2', '01'): 3, ('2', '05'):4}[key]
        new_task = 'localizer'
    elif task == 'visualmemory':
        run_ = hyperface_runs.index(int(run)) + 1
        with open(f'{timing_dir}/sub-{sid}_order_runs.json', 'r') as f:
            res = json.load(f)
        new_run = int(res[f'ses-{ses}_run-{run_}']['order_orig'])
        new_task = 'hyperface'
    elif ses == 'budapest' and task == 'movie':
        new_run = int(run)
        assert new_run <= 5
        new_task = 'budapest'
    else:
        raise ValueError

    label = f'{sid}_{new_task}_{new_run:02d}_{lr}h'
    return label


if __name__ == '__main__':
    dset = 'hyperface'
    fmriprep_version = '21.0.2'
    bids_dir = os.path.realpath(f'./bids_root/{dset}')
    n_procs = 40

    sids = sorted([os.path.basename(_)[4:] for _ in glob(os.path.join(bids_dir, 'sub-*'))])
    print(sids)

    idx = int(sys.argv[1])
    sid = sids[idx]
    print(sid)

    config = {
        'dset': dset,
        'sid': sid,
        'fmriprep_version': fmriprep_version,
        'n_procs': n_procs,

        'singularity_image': os.path.realpath(os.path.expanduser(f'~/lab/fmriprep_{fmriprep_version}.sif')),
        'singularity_home': os.path.realpath(os.path.expanduser(f'~/lab/singularity_home/fmriprep')),

        'bids_dir': bids_dir,
        'output_root': os.path.realpath(os.path.expanduser(f'~/singularity_home/data/{dset}_{fmriprep_version}')),
        'fmriprep_out': os.path.realpath(f'./fmriprep_out_root/{dset}_{fmriprep_version}/output_{sid}'),
        'fmriprep_work': os.path.realpath(f'./fmriprep_work_root/{dset}_{fmriprep_version}/work_{sid}'),

        'singularity_options': [
            '-B', '/dartfs:/dartfs',
            '-B', '/scratch:/scratch',
            '-B', '/dartfs-hpc:/dartfs-hpc',
        ],

        'fmriprep_options': [
            '--skull-strip-fixed-seed',
            '--omp-nthreads', '1',
            '--nprocs', str(n_procs),
            '--random-seed', '0',
            '--skip_bids_validation',
            '--ignore', 'slicetiming',
            '--use-syn-sdc',
            '--output-spaces', 'fsaverage5',
        ],

        'resample_flavor': 'on-avg-1031-final_ico32_2step_normals_equal',
    }

    run_fmriprep(config, cleanup=True)

    out_dir = os.path.expanduser(f'~/singularity_home/data/haxbylab-siemens_{fmriprep_version}/{config["resample_flavor"]}/no-gsr')
    regression_workflow(config, out_dir, rename_func, n_jobs=n_procs)

import os
import sys
from glob import glob
from process.fmriprep import run_fmriprep
from process.regression import regression_workflow


def rename_func(fn):
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

    if ses == 'movie' and task == 'movie':
        new_run = int(run)
        new_task = 'forrest'
    elif task in ['objectcategories', 'retmapccw', 'retmapclw', 'retmapcon', 'retmapexp', 'movielocalizer']:
        new_run = int(run)
        new_task = task
    else:
        raise ValueError

    label = f'{sid}_{new_task}_{new_run:02d}_{lr}h'
    return label



if __name__ == '__main__':
    dset = 'forrest'
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
            '--bids-filter-file', os.path.expanduser('~/github/process/scripts/fmriprep_forrest_filter.json'),
            '--use-syn-sdc',
            '--output-spaces', 'fsaverage5',
        ],

        'resample_flavor': 'on-avg-1031-final_ico32_2step_normals_equal',
    }

    run_fmriprep(config, cleanup=True)

    out_dir = os.path.expanduser(f'~/singularity_home/data/final/forrest_{fmriprep_version}/onavg-ico32/2step_normals_equal/no-gsr')
    regression_workflow(config, out_dir, rename_func, n_jobs=n_procs, resample_flavor='on-avg-1031-final_ico32_2step_normals_equal', ignore_non_existing=True)
    out_dir = os.path.expanduser(f'~/singularity_home/data/final/forrest_{fmriprep_version}/fsavg-ico32/2step_normals_equal/no-gsr')
    regression_workflow(config, out_dir, rename_func, n_jobs=n_procs, resample_flavor='fsaverage_ico32_2step_normals_equal', ignore_non_existing=True)

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
        if key not in ['sub', 'task', 'run', 'lr']:
            raise ValueError
    sid, task, run, lr = info['sub'], info['task'], info['run'], info['lr']

    if task == 'video':
        new_task = 'whiplash'
    elif task == 'localizer':
        new_task = task
    else:
        raise ValueError
    new_run = int(run)

    label = f'{sid}_{new_task}_{new_run:02d}_{lr}h'
    return label


if __name__ == '__main__':
    dset = 'whiplash-jane'
    fmriprep_version = '21.0.2'
    bids_dir = os.path.realpath(os.path.expanduser(f'~/lab/BIDS/{dset}'))
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

    }

    combinations = []
    for space in ['fsavg-ico32', 'onavg-ico32', 'onavg-ico64']:
        combinations.append((space, '1step_pial_area'))
    config['combinations'] = combinations

    run_fmriprep(config, cleanup=False)

    for space, resample_flavor in combinations:
        out_dir = os.path.expanduser(f'~/singularity_home/data/final/forrest_{fmriprep_version}/{space}/{resample_flavor}/no-gsr')
        print(out_dir)
        regression_workflow(config, out_dir, rename_func, space, resample_flavor, n_jobs=n_procs, ignore_non_existing=True)

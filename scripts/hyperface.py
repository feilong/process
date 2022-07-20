import os
import sys
from glob import glob
from process.fmriprep import run_fmriprep


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
    }

    run_fmriprep(config, cleanup=True)

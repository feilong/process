import os
import sys
from glob import glob
from process.fmriprep import run_fmriprep


if __name__ == '__main__':
    dset = 'ds003233'
    fmriprep_version = '22.0.0rc0'
    n_procs = 40
    config = {
        'dset': dset,
        'fmriprep_version': fmriprep_version,

        'singularity_image': os.path.realpath(os.path.expanduser(f'~/lab/fmriprep_{fmriprep_version}.sif')),
        'singularity_home': os.path.realpath(os.path.expanduser(f'~/lab/singularity_home/fmriprep')),

        'bids_dir': os.path.realpath(f'./bids_root/{dset}'),
        'output_root': os.path.realpath(os.path.expanduser(f'~/singularity_home/data/{dset}_{fmriprep_version}')),
        'fmriprep_out': os.path.realpath(f'./fmriprep_out_root/{dset}_{fmriprep_version}/output'),
        'fmriprep_work': os.path.realpath(f'./fmriprep_work_root/{dset}_{fmriprep_version}/work'),

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
            '--output-spaces', 'fsaverage', 'MNI152NLin2009cAsym:res-2',
        ],
    }

    print(config['bids_dir'])
    sids = sorted([os.path.basename(_)[4:] for _ in glob(os.path.join(config['bids_dir'], 'sub-*'))])
    print(sids)

    idx = int(sys.argv[1])
    config['sid'] = sids[idx]
    run_fmriprep(config)

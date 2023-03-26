import os
import sys
from glob import glob
from process.main import PreprocessWorkflow
from process.fmriprep import fmriprep_success


if __name__ == '__main__':
    dset = 'dalmatians'
    fmriprep_version = '20.2.7'
    bids_dir = os.path.realpath(os.path.expanduser(f'~/lab/BIDS/101_Dalmatians'))
    n_procs = 40 if os.uname()[1].startswith('ndoli') else int(os.environ['SLURM_CPUS_PER_TASK'])

    sids = sorted([os.path.basename(_)[4:] for _ in glob(os.path.join(bids_dir, 'sub-*'))])
    print(sids)

    idx = int(sys.argv[1])
    sid = sids[idx]
    print(idx, sid)

    config = {
        'dset': dset,
        'sid': sid,
        'fmriprep_version': fmriprep_version,
        'n_procs': n_procs,

        'singularity_image': os.path.realpath(os.path.expanduser(f'~/lab/fmriprep_{fmriprep_version}.sif')),
        'singularity_home': os.path.realpath(os.path.expanduser(f'~/lab/singularity_home/fmriprep')),

        'bids_dir': bids_dir,
        'output_root': os.path.realpath(os.path.expanduser(f'~/lab/nb-data-archive/{dset}/{fmriprep_version}')),
        'output_data_root': os.path.realpath(os.path.expanduser(f'~/lab/nb-data/{dset}/{fmriprep_version}')),
        'output_summary_root': os.path.realpath(os.path.expanduser(f'~/lab/nb-data-summary/{dset}/{fmriprep_version}')),
        'fmriprep_out': os.path.realpath(os.path.expanduser(f'~/lab/fmriprep_out_root/{dset}_{fmriprep_version}/output_{sid}')),
        'fmriprep_work': os.path.realpath(os.path.expanduser(f'~/lab/fmriprep_work_root/{dset}_{fmriprep_version}/work_{sid}')),

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
            '--output-spaces', 'fsaverage5', 'MNI152NLin2009cAsym:res-1',
        ],
    }
    if not os.uname()[1].startswith('ndoli'):
        config['fmriprep_options'] += ['--mem_mb', str(8000*n_procs)]

    combinations = []
    for space in ['fsavg-ico32', 'onavg-ico32', 'onavg-ico48', 'onavg-ico64']:
        combinations.append((space, '1step_pial_area'))
        combinations.append((space, '1step_pial_overlap'))
    config['combinations'] = combinations.copy()
    config['combinations'].append(('fsavg-ico32', '2step_normals-sine_nnfr'))
    config['combinations'].append(('fsavg-ico32', '2step_normals-equal_nnfr'))

    wf = PreprocessWorkflow(config)
    # if not fmriprep_success(1, os.path.join(wf.log_dir, f'{wf.sid}_fmriprep_stdout.txt'), wf.fmriprep_out):
    wf.fmriprep(anat_only=True)
    wf.xform()
    # wf.resample()
    # wf.compress()
    # wf.confound()

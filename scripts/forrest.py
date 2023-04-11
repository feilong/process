import os
import sys
from glob import glob
from process.main import PreprocessWorkflow


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
    # fmriprep_version = '22.1.0-fix'
    fmriprep_version = '20.2.7'
    bids_dir = os.path.realpath(os.path.expanduser(f'~/lab/BIDS/ds000113'))
    n_procs = 40 if os.uname()[1].startswith('ndoli') else int(os.environ['SLURM_CPUS_PER_TASK'])

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
        'output_root': os.path.realpath(os.path.expanduser(f'~/lab/nb-data-archive/{dset}_{fmriprep_version}')),
        'output_data_root': os.path.realpath(os.path.expanduser(f'~/lab/nb-data/{dset}_{fmriprep_version}')),
        'output_summary_root': os.path.realpath(os.path.expanduser(f'~/lab/nb-data-summary/{dset}_{fmriprep_version}')),
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
            '--bids-filter-file', os.path.expanduser('~/github/process/scripts/fmriprep_forrest_filter.json'),
            '--use-syn-sdc',
            '--output-spaces', 'fsaverage5', 'MNI152NLin2009cAsym:res-1',
        ],
    }

    if not os.uname()[1].startswith('ndoli'):
        config['fmriprep_options'] += ['--mem_mb', str(8000*n_procs)]

    combinations = []
    for space in ['fsavg-ico32', 'onavg-ico32', 'onavg-ico48', 'onavg-ico64']:
        combinations.append((space, '1step_pial_area'))
    config['combinations'] = combinations.copy()
    config['combinations'].append(('fsavg-ico32', '2step_normals-sine_nnfr'))
    config['combinations'].append(('fsavg-ico32', '2step_normals-equal_nnfr'))

    # removed from filter:
    #  "fmap": {"datatype": "fmap"},

    wf = PreprocessWorkflow(config)
    # if not fmriprep_success(1, os.path.join(wf.log_dir, f'{wf.sid}_fmriprep_stdout.txt'), wf.fmriprep_out):
    #     wf.fmriprep()
    assert wf.fmriprep()
    wf.resample(filter_=lambda fns: [_ for _ in fns if 'ses-localizer' in _ or 'ses-movie' in _])
    wf.compress()
    wf.confound()


    # run_fmriprep(config, cleanup=False)

    # run_fmriprep(config, cleanup=True)

    # combinations = []
    # for space in ['fsavg-ico32', 'fsavg-ico64', 'fslr-ico32', 'fslr-ico64', 'onavg-ico32', 'onavg-ico64']:
    #     combinations.append((space, '1step_pial_area'))
    # for step in ['1step', '2step']:
    #     for projection_type in ['normals_equal', 'pial']:
    #         for resample_method in ['nnfr', 'area']:
    #             flavor = f'{step}_{projection_type}_{resample_method}'
    #             key = ('onavg-ico64', flavor)
    #             if key not in combinations:
    #                 combinations.append(key)

    # for space, resample_flavor in combinations:
    #     out_dir = os.path.expanduser(f'~/singularity_home/data/final/forrest_{fmriprep_version}/{space}/{resample_flavor}/no-gsr')
    #     print(out_dir)
    #     regression_workflow(config, out_dir, rename_func, space, resample_flavor, n_jobs=n_procs, ignore_non_existing=True)

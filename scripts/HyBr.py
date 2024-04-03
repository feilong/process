import os
import sys
from glob import glob
from process.main import PreprocessWorkflow



if __name__ == '__main__':
    dset = 'hybr'
    fmriprep_version = '20.2.7'
    bids_dir = os.path.realpath(os.path.expanduser(f'/dartfs/rc/lab/H/HaxbyLab/xiaoxuan/pilot/Haxby/Xiaoxuan/1110_HyperBasePilot/'))
    n_procs = 40 if os.uname()[1].startswith('ndoli') else int(os.environ['SLURM_CPUS_PER_TASK'])

    sids = sorted([os.path.basename(_)[4:] for _ in glob(os.path.join(bids_dir, 'sub-*'))])
    # sids = ['01', '02', '03', '04', '05', '06', '09', '10', '14', '15', '16', '17', '18', '19', '20']
    print(sids)

    idx = int(sys.argv[1])
    sid = sids[idx]
    print(sid)

    config = {
        'dset': dset,
        'sid': sid,
        'fmriprep_version': fmriprep_version,
        'n_procs': n_procs,

        'singularity_image': os.path.realpath(os.path.expanduser(f'/dartfs/rc/lab/H/HaxbyLab/feilong/fmriprep_{fmriprep_version}.sif')),
        'singularity_home': os.path.realpath(os.path.expanduser(f'/dartfs/rc/lab/H/HaxbyLab/feilong/singularity_home/fmriprep')),

        'bids_dir': bids_dir,
        'output_root': os.path.realpath(os.path.expanduser(f'/dartfs/rc/lab/H/HaxbyLab/xiaoxuan/{dset}-archive/{fmriprep_version}')),
        'output_data_root': os.path.realpath(os.path.expanduser(f'/dartfs/rc/lab/H/HaxbyLab/xiaoxuan/{dset}/{fmriprep_version}')),
        'output_summary_root': os.path.realpath(os.path.expanduser(f'/dartfs/rc/lab/H/HaxbyLab/xiaoxuan/{dset}-summary/{fmriprep_version}')),
        'fmriprep_out': os.path.realpath(os.path.expanduser(f'/dartfs/rc/lab/H/HaxbyLab/xiaoxuan/fmriprep_out_root/{dset}_{fmriprep_version}/output_{sid}')),
        'fmriprep_work': os.path.realpath(os.path.expanduser(f'/dartfs/rc/lab/H/HaxbyLab/xiaoxuan/fmriprep_work_root/{dset}_{fmriprep_version}/work_{sid}')),

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
    config['combinations'].append(('onavg-ico32', '2step_normals-sine_nnfr'))
    config['combinations'].append(('onavg-ico32', '2step_normals-equal_nnfr'))


    wf = PreprocessWorkflow(config)
    assert wf.fmriprep(anat_only=True)
    assert wf.xform()
    assert wf.anatomy()
    assert wf.fmriprep()
    assert wf.confound()
    assert wf.compress()
    assert wf.archive()
    assert wf.resample()
    assert wf.cleanup()

    # wf.unpack(filter_=filter_)
    # assert wf.resample(filter_=filter_)

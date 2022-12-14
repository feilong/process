import os
import subprocess
import shutil
from glob import glob

from process.compression import copy_files_to_lzma_tar
from process.surface import resample_workflow


def fmriprep_cmd(config):
    cmd = [
        'singularity', 'run', '-e',
        *config['singularity_options'], '-H', config['singularity_home'],
        config['singularity_image'],
        '--participant-label', config['sid'],
        *config['fmriprep_options'],
        '--fs-license-file',  config['singularity_home'] + '/FS_license.txt',
        '-w', config['fmriprep_work'],
        config['bids_dir'], config['fmriprep_out'], 'participant',
    ]
    return cmd


def run_fmriprep(config, cleanup=True):
    sid = config['sid']

    for key in ['fmriprep_work', 'fmriprep_out', 'output_root']:
        os.makedirs(config[key], exist_ok=True)

    log_dir = os.path.join(config['output_root'], 'logs')
    fmriprep_dir = os.path.join(config['output_root'], 'fmriprep')
    freesurfer_dir = os.path.join(config['output_root'], 'freesurfer')
    summary_dir = os.path.join(config['output_root'], 'summary')
    confounds_dir = os.path.join(config['output_root'], 'confounds')
    resample_dir = os.path.join(config['output_root'], 'resampled')
    for dir_name in [log_dir, fmriprep_dir, freesurfer_dir, summary_dir, confounds_dir]:
        os.makedirs(dir_name, exist_ok=True)

    finish_fn = f'{log_dir}/{sid}_fmriprep_finish.txt'
    running_fn = f'{log_dir}/{sid}_fmriprep_running.txt'
    if os.path.exists(finish_fn):
        return
    if os.path.exists(running_fn):
        return
    with open(running_fn, 'w') as f:
        f.write('')

    fmriprep_fn = os.path.join(fmriprep_dir, f'{sid}.tar.lzma')
    freesurfer_fn = os.path.join(freesurfer_dir, f'{sid}.tar.lzma')
    summary_fn = os.path.join(summary_dir, f'{sid}.tar.lzma')
    confounds_fn = os.path.join(confounds_dir, f'{sid}.tar.lzma')
    if all([os.path.exists(_) for _ in [fmriprep_fn, freesurfer_fn, summary_fn, confounds_fn, resample_dir]]):
        return

    fmriprep_out = os.path.join(config['fmriprep_out'], f'sub-{sid}')
    freesurfer_out = os.path.join(config['fmriprep_out'], 'sourcedata', 'freesurfer', f'sub-{sid}')
    major, minor = config['fmriprep_version'].split('.')[:2]
    # work_out = os.path.join(config['fmriprep_work'], f'fmriprep_{major}_{minor}_wf', f'single_subject_{sid}_wf')
    work_out = os.path.join(config['fmriprep_work'], f'fmriprep_wf', f'single_subject_{sid}_wf')

    if not all([os.path.exists(_) for _ in [fmriprep_out, freesurfer_out, work_out]]):
        cmd = fmriprep_cmd(config)
        stdout_fn = os.path.join(log_dir, f'{sid}_fmriprep_stdout.txt')
        stderr_fn = os.path.join(log_dir, f'{sid}_fmriprep_stderr.txt')
        with open(stdout_fn, 'w') as f1, open(stderr_fn, 'w') as f2:
            proc = subprocess.run(cmd, stdout=f1, stderr=f2)
        with open(stdout_fn, 'r') as f:
            content = f.read()
        if proc.returncode != 0 and 'fMRIPrep finished successfully!' not in content:
            print(config)
            print(cmd)
            print(config['dset'], sid, proc.returncode)
            return

    resample_workflow(
        sid=sid, bids_dir=config['bids_dir'], fs_dir=freesurfer_out, wf_root=work_out, out_dir=resample_dir,
        n_jobs=config['n_procs'], combinations=config['combinations'])

    copy_files_to_lzma_tar(
        fmriprep_fn,
        # [fmriprep_out, fmriprep_out + '.html'],
        [_ for _ in sorted(glob(os.path.join(config['fmriprep_out'], '*'))) if os.path.basename(_) != 'sourcedata'],
        rename_func=lambda x: os.path.relpath(x, config['fmriprep_out'])
    )
    copy_files_to_lzma_tar(
        freesurfer_fn,
        [freesurfer_out],
        rename_func=lambda x: os.path.relpath(x, os.path.join(config['fmriprep_out'], 'sourcedata', 'freesurfer'))
    )
    copy_files_to_lzma_tar(
        summary_fn,
        [fmriprep_out + '.html'] + sorted(glob(os.path.join(fmriprep_out, 'figures', '*'))),
        rename_func=lambda x: os.path.relpath(x, config['fmriprep_out']),
    )
    copy_files_to_lzma_tar(
        confounds_fn,
        sorted(glob(os.path.join(fmriprep_out, 'func', '*.tsv'))) + sorted(glob(os.path.join(fmriprep_out, 'ses-*', 'func', '*.tsv'))),
        rename_func=lambda x: os.path.relpath(x, config['fmriprep_out']),
    )

    if cleanup:
        if all([os.path.exists(_) for _ in [fmriprep_fn, freesurfer_fn, summary_fn, confounds_fn]]):
            for root in [config['fmriprep_out'], config['fmriprep_work']]:
                shutil.rmtree(root)

    with open(finish_fn, 'w') as f:
        f.write('')
    if os.path.exists(running_fn):
        os.remove(running_fn)

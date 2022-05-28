import os
import subprocess
import shutil


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
    for dir_name in [log_dir, fmriprep_dir, freesurfer_dir, summary_dir]:
        os.makedirs(dir_name, exist_ok=True)

    finish_fn = f'{log_dir}/{sid}_fmriprep_finish.txt'
    running_fn = f'{log_dir}/{sid}_fmriprep_running.txt'
    if os.path.exists(finish_fn):
        return
    if os.path.exists(running_fn):
        return
    with open(running_fn, 'w') as f:
        f.write('')

    fmriprep_fn = f'{fmriprep_dir}/{sid}.tar.lzma'
    freesurfer_fn = f'{fmriprep_dir}/{sid}.tar.lzma'
    if all([os.path.exists(_) for _ in [fmriprep_fn, freesurfer_fn]]):
        return

    fmriprep_out = os.path.join(config['fmriprep_out'], f'sub-{sid}')
    freesurfer_out = os.path.join(config['fmriprep_out'], 'sourcedata', 'freesurfer', f'sub-{sid}')
    work_out = os.path.join(config['fmriprep_out'], 'fmriprep_wf', f'single_subject_{sid}_wf')

    if not all([os.path.exists(_) for _ in [fmriprep_out, freesurfer_out, work_out]]):
        cmd = fmriprep_cmd(config)
        stdout_fn = os.path.join(log_dir, f'{sid}_fmriprep_stdout.txt')
        stderr_fn = os.path.join(log_dir, f'{sid}_fmriprep_stderr.txt')
        with open(stdout_fn, 'w') as f1, open(stderr_fn, 'w') as f2:
            proc = subprocess.run(cmd, stdout=f1, stderr=f2)
        if proc.returncode != 0:
            print(config)
            print(cmd)
            print(config['dset'], sid, proc.returncode)
            return

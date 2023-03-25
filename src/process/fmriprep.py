import os
from glob import glob


def fmriprep_cmd(config, *additional_options):
    cmd = [
        'singularity', 'run', '-e',
        *config['singularity_options'], '-H', config['singularity_home'],
        config['singularity_image'],
        '--participant-label', config['sid'],
        *config['fmriprep_options'],
        *additional_options,
        '--fs-license-file',  config['singularity_home'] + '/FS_license.txt',
        '-w', config['fmriprep_work'],
        config['bids_dir'], config['fmriprep_out'], 'participant',
    ]
    return cmd


def fmriprep_success(returncode, stdout_fn, fmriprep_out):
    if returncode == 0:
        return True

    if os.path.exists(stdout_fn):
        with open(stdout_fn, 'r') as f:
            content = f.read()
        if 'fMRIPrep finished successfully!' in content:
            return True

        crashes = glob(os.path.join(fmriprep_out, 'log', '*', 'crash-*.txt'))
        if len(crashes) <= 2 and all(['_midthickness' in _ for _ in crashes]):
            return True

    return False

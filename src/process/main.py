import os
import subprocess
import shutil
from glob import glob

from .fmriprep import fmriprep_cmd
from .compression import copy_files_to_lzma_tar
from .resample_workflow import resample_workflow


class PreprocessWorkflow(object):
    def __init__(self, config):
        self.config = config
        sid = self.config['sid']
        self.sid = sid

        for key in ['fmriprep_work', 'fmriprep_out', 'output_root']:
            os.makedirs(self.config[key], exist_ok=True)

        self.log_dir = os.path.join(self.config['output_root'], 'logs')
        self.fmriprep_dir = os.path.join(self.config['output_root'], 'fmriprep')
        self.freesurfer_dir = os.path.join(self.config['output_root'], 'freesurfer')
        self.summary_dir = os.path.join(self.config['output_summary_root'], 'summary')
        self.confounds_dir = os.path.join(self.config['output_data_root'], 'confounds')
        self.resample_dir = os.path.join(self.config['output_data_root'], 'resampled')
        for dir_name in [self.log_dir, self.fmriprep_dir, self.freesurfer_dir, self.summary_dir, self.confounds_dir]:
            os.makedirs(dir_name, exist_ok=True)

        self.fmriprep_out = os.path.join(self.config['fmriprep_out'], f'sub-{sid}')
        self.freesurfer_out = os.path.join(self.config['fmriprep_out'], 'sourcedata', 'freesurfer', f'sub-{sid}')
        major, minor = self.config['fmriprep_version'].split('.')[:2]
        if int(major) >= 22:
            self.work_out = os.path.join(config['fmriprep_work'], f'fmriprep_{major}_{minor}_wf', f'single_subject_{sid}_wf')
        else:
            self.work_out = os.path.join(config['fmriprep_work'], f'fmriprep_wf', f'single_subject_{sid}_wf')

        self.fmriprep_fn = os.path.join(self.fmriprep_dir, f'{sid}.tar.lzma')
        self.freesurfer_fn = os.path.join(self.freesurfer_dir, f'{sid}.tar.lzma')
        self.summary_fn = os.path.join(self.summary_dir, f'{sid}.tar.lzma')
        self.confounds_fn = os.path.join(self.confounds_dir, f'{sid}.tar.lzma')
    
    def _run_method(self, name='fmriprep'):
        sid = self.sid
        finish_fn = f'{self.log_dir}/{sid}_{name}_finish.txt'
        running_fn = f'{self.log_dir}/{sid}_{name}_running.txt'
        error_fn = f'{self.log_dir}/{sid}_{name}_error.txt'

        if os.path.exists(finish_fn):
            return True
        if os.path.exists(error_fn):
            return False
        if os.path.exists(running_fn):
            return False
        with open(running_fn, 'w') as f:
            f.write('')

        if name == 'fmriprep':
            step = self._run_fmriprep
        elif name == 'resample':
            step = self._run_resample
        elif name == 'compress':
            step = self._run_compress
        elif name == 'cleanup':
            step = self._run_cleanup
        else:
            raise ValueError
        try:
            success, message = step()
        except Exception as e:
            success = False
            message = str(e)

        fn = finish_fn if success else error_fn
        with open(fn, 'w') as f:
            f.write(message)
        if os.path.exists(running_fn):
            os.remove(running_fn)

        if not success:
            print(message)
            exit(1)

    def _run_fmriprep(self):
        sid = self.sid
        cmd = fmriprep_cmd(self.config)
        stdout_fn = os.path.join(self.log_dir, f'{sid}_fmriprep_stdout.txt')
        stderr_fn = os.path.join(self.log_dir, f'{sid}_fmriprep_stderr.txt')
        with open(stdout_fn, 'w') as f1, open(stderr_fn, 'w') as f2:
            proc = subprocess.run(cmd, stdout=f1, stderr=f2)
        with open(stdout_fn, 'r') as f:
            content = f.read()
            success = not (
                proc.returncode != 0 and 'fMRIPrep finished successfully!' not in content)
        message = '\n'.join([
            f"{self.config['dset']}, {sid}, {proc.returncode}",
            str(self.config), str(cmd), ' '.join(cmd)])
        return success, message

    def _run_resample(self):
        resample_workflow(
            sid=self.sid, bids_dir=self.config['bids_dir'],
            fs_dir=self.freesurfer_out, wf_root=self.work_out, out_dir=self.resample_dir,
            n_jobs=self.config['n_procs'], combinations=self.config['combinations'])
        return True, ''

    def _run_compress(self):
        copy_files_to_lzma_tar(
            self.fmriprep_fn,
            [_ for _ in sorted(glob(os.path.join(self.config['fmriprep_out'], '*'))) if os.path.basename(_) != 'sourcedata'],
            rename_func=lambda x: os.path.relpath(x, config['fmriprep_out']),
            exclude = lambda fn: fn.endswith('space-MNI152NLin2009cAsym_res-1_desc-preproc_bold.nii.gz')
        )
        copy_files_to_lzma_tar(
            self.freesurfer_fn,
            [self.freesurfer_out],
            rename_func=lambda x: os.path.relpath(x, os.path.join(self.config['fmriprep_out'], 'sourcedata', 'freesurfer'))
        )
        copy_files_to_lzma_tar(
            self.summary_fn,
            [self.fmriprep_out + '.html'] + sorted(glob(os.path.join(self.fmriprep_out, 'figures', '*'))),
            rename_func=lambda x: os.path.relpath(x, self.config['fmriprep_out']),
        )
        copy_files_to_lzma_tar(
            self.confounds_fn,
            sorted(glob(os.path.join(self.fmriprep_out, 'func', '*.tsv'))) + sorted(glob(os.path.join(self.fmriprep_out, 'ses-*', 'func', '*.tsv'))),
            rename_func=lambda x: os.path.relpath(x, self.config['fmriprep_out']),
        )
        return True, ''

    def _cleanup(self):
        if all([os.path.exists(_) for _ in [self.fmriprep_fn, self.freesurfer_fn, self.summary_fn, self.confounds_fn]]):
            for root in [self.config['fmriprep_out'], self.config['fmriprep_work']]:
                if os.path.exists(root):
                    shutil.rmtree(root)
            return True, ''
        else:
            return False, 'Not all output files exist.'

    def fmriprep(self):
        self._run_method('fmriprep')

    def resample(self):
        self._run_method('resample')

    def compress(self):
        self._run_method('compress')

    def cleanup(self):
        self._run_method('cleanup')

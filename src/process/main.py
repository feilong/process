import os
import subprocess
import shutil
from glob import glob

from .fmriprep import fmriprep_cmd, fmriprep_success
from .compression import copy_files_to_lzma_tar
from .resample_workflow import resample_workflow
from .confound import confound_workflow
from .surface import xform_workflow
from .archive import archive_subject_work_dir
from .anatomy import run_freesurfer_invivo_v1, resample_freesurfer
from .t2star import t2smap_cmd


class PreprocessWorkflow(object):
    def __init__(self, config):
        self.config = config
        sid = self.config['sid']
        self.sid = sid

        for key in ['fmriprep_work', 'fmriprep_out', 'output_root']:
            os.makedirs(self.config[key], exist_ok=True)

        self.log_dir = os.path.join(self.config['output_root'], 'logs')

        fmriprep_dir = os.path.join(self.config['output_root'], 'fmriprep')
        freesurfer_dir = os.path.join(self.config['output_root'], 'freesurfer')
        summary_dir = self.config['output_summary_root']
        confounds_dir = os.path.join(self.config['output_root'], 'confounds')
        self.resample_dir = os.path.join(self.config['output_data_root'], 'resampled')
        self.xform_dir = os.path.join(self.config['output_data_root'], 'xforms')
        self.anat_dir = os.path.join(self.config['output_data_root'], 'anatomy')
        self.confound_dir = os.path.join(self.config['output_data_root'], 'confounds')
        for dir_name in [self.log_dir, fmriprep_dir, freesurfer_dir, summary_dir, confounds_dir, self.confound_dir]:
            os.makedirs(dir_name, exist_ok=True)

        major, minor = self.config['fmriprep_version'].split('.')[:2]
        if int(major) >= 22:
            self.work_out = os.path.join(config['fmriprep_work'], f'fmriprep_{major}_{minor}_wf', f'single_subject_{sid}_wf')
        else:
            self.work_out = os.path.join(config['fmriprep_work'], f'fmriprep_wf', f'single_subject_{sid}_wf')
        if int(major) >= 21:
            self.freesurfer_out = os.path.join(self.config['fmriprep_out'], 'sourcedata', 'freesurfer', f'sub-{sid}')
            self.fmriprep_out = os.path.join(self.config['fmriprep_out'], f'sub-{sid}')
        else:
            self.freesurfer_out = os.path.join(self.config['fmriprep_out'], 'freesurfer', f'sub-{sid}')
            self.fmriprep_out = os.path.join(self.config['fmriprep_out'], 'fmriprep', f'sub-{sid}')

        self.fmriprep_fn = os.path.join(fmriprep_dir, f'{sid}.tar.lzma')
        self.freesurfer_fn = os.path.join(freesurfer_dir, f'{sid}.tar.lzma')
        self.summary_fn = os.path.join(summary_dir, f'{sid}.tar.lzma')
        self.confounds_fn = os.path.join(confounds_dir, f'{sid}.tar.lzma')
    
    def _run_method(self, name='fmriprep', log_name=None, **kwargs):
        sid = self.sid
        log_name = name if log_name is None else log_name
        finish_fn = f'{self.log_dir}/{sid}_{log_name}_finish.txt'
        running_fn = f'{self.log_dir}/{sid}_{log_name}_running.txt'
        error_fn = f'{self.log_dir}/{sid}_{log_name}_error.txt'

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
        elif name == 'xform':
            step = self._run_xform
        elif name == 'resample':
            step = self._run_resample
        elif name == 'compress':
            step = self._run_compress
        elif name == 'cleanup':
            step = self._run_cleanup
        elif name == 'confound':
            step = self._run_confound
        elif name == 'archive':
            step = self._run_archive
        elif name == 'anatomy':
            step = self._run_anatomy
        elif name == 't2star':
            step = self._run_t2star
        else:
            raise ValueError
        try:
            success, message = step(**kwargs)
            error = None
        except Exception as e:
            success = False
            error = e
            message = str(e)

        fn = finish_fn if success else error_fn
        with open(fn, 'w') as f:
            f.write(message)
        if os.path.exists(running_fn):
            os.remove(running_fn)

        if error is not None:
            raise error

        if not success:
            print(message)
            exit(1)
        return success

    def _run_fmriprep(self, anat_only=False, log_name=None, additional_options=None):
        sid = self.sid
        if additional_options is None:
            additional_options = []
        additional_options += ['--anat-only'] if anat_only else []
        cmd = fmriprep_cmd(self.config, *additional_options)
        if log_name is None:
            log_name = 'anatonly' if anat_only else 'fmriprep'
        stdout_fn = os.path.join(self.log_dir, f'{sid}_{log_name}_stdout.txt')
        stderr_fn = os.path.join(self.log_dir, f'{sid}_{log_name}_stderr.txt')
        with open(stdout_fn, 'w') as f1, open(stderr_fn, 'w') as f2:
            proc = subprocess.run(cmd, stdout=f1, stderr=f2)

        success = fmriprep_success(proc.returncode, stdout_fn, self.fmriprep_out)

        message = '\n'.join([
            f"{self.config['dset']}, {sid}, {proc.returncode}",
            str(self.config), str(cmd), ' '.join(cmd)])
        return success, message

    def _run_t2star(self):
        sid = self.sid
        bids_dir = self.config['bids_dir']
        wf_root = self.work_out

        raw_bolds = sorted(glob(f'{bids_dir}/sub-{sid}/ses-*/func/*_echo-1_bold.nii.gz')) + \
            sorted(glob(f'{bids_dir}/sub-{sid}/func/*_echo-1_bold.nii.gz'))
        labels = [os.path.basename(_).split(f'sub-{sid}_', 1)[1].rsplit('_bold.nii.gz', 1)[0] for _ in raw_bolds]

        success = True
        for label in labels:
            label2 = label.replace('-', '_')
            wf_dir = (f'{wf_root}/func_preproc_{label2}_wf')
            cmd = t2smap_cmd(self.config, wf_dir)
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                success = False
                print(proc.stdout)
                print(proc.stderr)
        return success, ''

    def _run_anatomy(self):
        proc = run_freesurfer_invivo_v1(self.config, self.log_dir)
        if proc.returncode == 0:
            resample_freesurfer(self.config, self.anat_dir, self.xform_dir)
            return True, ''
        else:
            return False, ''

    def _run_xform(self):
        xform_workflow(
            self.sid, fs_dir=self.freesurfer_out, xform_dir=self.xform_dir,
            combinations=self.config['combinations'])
        return True, ''

    def _run_resample(self, filter_):
        resample_workflow(
            sid=self.sid, bids_dir=self.config['bids_dir'],
            fs_dir=self.freesurfer_out, wf_root=self.work_out, out_dir=self.resample_dir, xform_dir=self.xform_dir,
            n_jobs=self.config['n_procs'], combinations=self.config['combinations'], filter_=filter_)
        return True, ''

    def _run_confound(self):
        confound_workflow(self.fmriprep_out, self.confound_dir)
        return True, ''

    def _run_compress(self):
        copy_files_to_lzma_tar(
            self.fmriprep_fn,
            [_ for _ in sorted(glob(os.path.join(self.config['fmriprep_out'], '*'))) if os.path.basename(_) != 'sourcedata'],
            rename_func=lambda x: os.path.relpath(x, os.path.dirname(self.fmriprep_out)),
            # rename_func=lambda x: os.path.relpath(x, self.config['fmriprep_out']),
            exclude = lambda fn: fn.endswith('space-MNI152NLin2009cAsym_res-1_desc-preproc_bold.nii.gz')
        )
        copy_files_to_lzma_tar(
            self.freesurfer_fn,
            [self.freesurfer_out],
            rename_func=lambda x: os.path.relpath(x, os.path.dirname(self.freesurfer_out)),
            # os.path.join(self.config['fmriprep_out'], 'sourcedata', 'freesurfer')
        )
        copy_files_to_lzma_tar(
            self.summary_fn,
            [self.fmriprep_out + '.html'] + sorted(glob(os.path.join(self.fmriprep_out, 'figures', '*'))),
            # rename_func=lambda x: os.path.relpath(x, self.config['fmriprep_out']),
            rename_func=lambda x: os.path.relpath(x, os.path.dirname(self.fmriprep_out)),
        )
        copy_files_to_lzma_tar(
            self.confounds_fn,
            sorted(glob(os.path.join(self.fmriprep_out, 'func', '*.tsv'))) + sorted(glob(os.path.join(self.fmriprep_out, 'ses-*', 'func', '*.tsv'))),
            rename_func=lambda x: os.path.relpath(x, self.config['fmriprep_out']),
        )
        return True, ''

    def _run_archive(self, filter_=None):
        sid = self.sid
        bids_dir=self.config['bids_dir']
        raw_bolds = sorted(glob(f'{bids_dir}/sub-{sid}/ses-*/func/*_bold.nii.gz')) + \
            sorted(glob(f'{bids_dir}/sub-{sid}/func/*_bold.nii.gz'))
        if filter_ is not None:
            raw_bolds = filter_(raw_bolds)
        labels = [os.path.basename(_).split(f'sub-{sid}_', 1)[1].rsplit('_bold.nii.gz', 1)[0] for _ in raw_bolds]
        wf_root = self.work_out
        out_dir = os.path.join(self.config['output_root'], 'fp_work')
        archive_subject_work_dir(self.sid, labels, wf_root, out_dir)
        return True, ''

    def _run_cleanup(self):
        if all([os.path.exists(_) for _ in [self.fmriprep_fn, self.freesurfer_fn, self.summary_fn, self.confounds_fn]]):
            for root in [self.config['fmriprep_out'], self.config['fmriprep_work']]:
                if os.path.exists(root):
                    shutil.rmtree(root)
            return True, ''
        else:
            return False, 'Not all output files exist.'

    def fmriprep(self, log_name=None, **kwargs):
        if 'anat_only' in kwargs and kwargs['anat_only'] and log_name is None:
            log_name = 'anatonly'
        return self._run_method('fmriprep', log_name=log_name, **kwargs)

    def xform(self):
        return self._run_method('xform')

    def resample(self, filter_=None, log_name=None):
        return self._run_method('resample', filter_=filter_, log_name=log_name)

    def compress(self):
        return self._run_method('compress')

    def cleanup(self):
        return self._run_method('cleanup')

    def confound(self):
        return self._run_method('confound')

    def archive(self, filter_=None, log_name=None):
        return self._run_method('archive', filter_=filter_, log_name=log_name)

    def anatomy(self):
        return self._run_method('anatomy')

    def t2star(self):
        return self._run_method('t2star')

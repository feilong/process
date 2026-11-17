import json
import math
import os
import subprocess
import shutil
from glob import glob
import tarfile
from datetime import datetime
from joblib import Parallel, delayed

from .fmriprep import fmriprep_cmd, fmriprep_success, check_fmriprep_version, find_runs
from .compression import copy_files_to_lzma_tar
from .resample_workflow import resample_workflow
from .resample import resample_surfaces, check_outputs, check_sdc
from .confound import confound_workflow
from .events import events_workflow
from .transform import xform_workflow
from .archive import archive_subject_work_dir
from .anatomy import run_freesurfer_invivo_v1, resample_freesurfer
from .t2star import t2smap_cmd


def unpack_lzma_file(lzma_fn, root):
    os.makedirs(root, exist_ok=True)
    t0 = datetime.now()
    with tarfile.open(lzma_fn) as tf:
        tf.extractall(root)
    t1 = datetime.now()
    print(t1, t1 - t0, lzma_fn, os.path.getsize(lzma_fn))


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
        self.events_dir = os.path.join(self.config['output_data_root'], 'events')
        for dir_name in [self.log_dir, fmriprep_dir, freesurfer_dir, summary_dir, confounds_dir, self.confound_dir, self.events_dir]:
            os.makedirs(dir_name, exist_ok=True)

        major, minor = self.config['fmriprep_version'].split('.')[:2]
        if major.startswith('next') or int(major) >= 22:
            self.work_out = os.path.join(config['fmriprep_work'], f'fmriprep_{major}_{minor}_wf', f'single_subject_{sid}_wf')
        else:
            self.work_out = os.path.join(config['fmriprep_work'], f'fmriprep_wf', f'single_subject_{sid}_wf')
        if major.startswith('next') or int(major) >= 21:
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
        elif name == 'partial_cleanup':
            step = self._run_partial_cleanup
        elif name == 'confound':
            step = self._run_confound
        elif name == 'events':
            step = self._run_events
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

    def _run_xform(self, **kwargs):
        args = dict(fs_dir=self.freesurfer_out, xform_dir=self.xform_dir,
            combinations=self.config['combinations'])
        args.update(kwargs)
        xform_workflow(self.sid, **args)
        return True, ''

    def _run_resample(self, filter_, **kwargs):
        # One-step resampling from fMRIPrep's outputs (25.x), or the workflow
        # based on its working directory (older versions).
        # The paths of this workflow come from the configured version, so
        # outputs of another major.minor version are rejected.
        version = check_fmriprep_version(self.config['fmriprep_out'], self.config['fmriprep_version'])
        major = version.split('.')[0]
        if major.isdigit() and int(major) >= 25:
            args = dict(
                sid=self.sid, bids_dir=self.config['bids_dir'],
                fmriprep_dir=os.path.dirname(self.fmriprep_out), out_dir=self.resample_dir,
                xform_dir=self.xform_dir, coverage_dir=os.path.join(self.confound_dir, 'coverage'),
                combinations=self.config['combinations'], fs_dir=self.freesurfer_out,
                filter_=filter_, n_jobs=self.config['n_procs'])
            args.update(kwargs)
            resample_surfaces(**args)
            return True, ''

        args = dict(
            sid=self.sid, bids_dir=self.config['bids_dir'],
            fs_dir=self.freesurfer_out, wf_root=self.work_out, out_dir=self.resample_dir, xform_dir=self.xform_dir,
            n_jobs=self.config['n_procs'], combinations=self.config['combinations'], filter_=filter_)
        args.update(kwargs)
        resample_workflow(**args)
        return True, ''

    def _check(self, filter_=None, n_frames=40, **kwargs):
        """Build the output and SDC report without writing it or exiting.

        Select runs once and share them between both checks. Discovery
        failures return a report with no runs; output-file and per-run SDC
        failures are recorded while the remaining checks continue. Argument
        errors and unexpected failures outside those checks propagate.
        Version validation and JSON writing belong to :meth:`check`.
        """
        report = {'sid': self.sid, 'runs': {}, 'errors': [], 'warnings': []}
        errors, warns = report['errors'], report['warnings']
        args = dict(
            sid=self.sid, bids_dir=self.config['bids_dir'],
            fmriprep_dir=os.path.dirname(self.fmriprep_out), out_dir=self.resample_dir,
            xform_dir=self.xform_dir, coverage_dir=os.path.join(self.confound_dir, 'coverage'),
            combinations=self.config['combinations'], fs_dir=self.freesurfer_out, filter_=filter_)
        args.update(kwargs)
        # Settings of `resample` that do not affect which files exist
        for key in ['n_jobs', 'n_depths']:
            args.pop(key, None)
        unknown = sorted(set(args) - {
            'sid', 'bids_dir', 'fmriprep_dir', 'out_dir', 'xform_dir', 'coverage_dir', 'combinations',
            'fs_dir', 'filter_', 'no_fieldmap', 'fieldmap_prefix', 'low_signal', 'low_coverage'})
        if unknown:
            raise TypeError(f'Unsupported arguments for the check step: {unknown}')
        # The runs are selected once, for both the files and the SDC check
        try:
            runs = find_runs(
                args['fmriprep_dir'], args['bids_dir'], args['sid'], args.pop('filter_'),
                no_fieldmap=args.pop('no_fieldmap', ()), fieldmap_prefix=args.pop('fieldmap_prefix', None))
        except Exception as e:
            errors.append(f'cannot find the runs ({type(e).__name__}: {e})')
            return report
        outputs = check_outputs(runs=runs, **args)
        report['runs'] = outputs['runs']
        errors += outputs['errors']
        warns += outputs['warnings']

        for run in runs:
            label = run['label']
            if run['fieldmap'] is None:
                sdc = {'status': 'skipped'}
                warns.append(f'{label}: SDC not checked (no fieldmap)')
            else:
                try:
                    sdc = check_sdc(run, args['bids_dir'], n_frames)
                except Exception as e:
                    sdc = {}
                    errors.append(f'{label}: SDC check failed ({type(e).__name__}: {e})')
                finite = bool(sdc) and all(math.isfinite(_) for _ in sdc.values())
                sdc = {key: (value if math.isfinite(value) else None) for key, value in sdc.items()}
                if not finite:
                    sdc['status'] = 'inconclusive'
                    if sdc.keys() != {'status'}:
                        errors.append(f'{label}: SDC check has nonfinite values')
                elif sdc['r_sdc'] <= sdc['r_nosdc']:
                    sdc['status'] = 'not_improved'
                    warns.append(
                        f"{label}: SDC does not improve the correlation with fMRIPrep's reference "
                        f"({sdc['r_sdc']:.4f} with, {sdc['r_nosdc']:.4f} without)")
                else:
                    sdc['status'] = 'ok'
            report['runs'].setdefault(label, {})['sdc'] = sdc
        return report

    def _run_confound(self, filter_):
        confound_workflow(self.fmriprep_out, self.confound_dir, filter_=filter_)
        return True, ''

    def _run_events(self, filter_):
        events_workflow(self.config['bids_dir'], self.events_dir, filter_=filter_)
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

    def _run_partial_cleanup(self, filter_):
        sid, bids_dir = self.sid, self.config['bids_dir']
        raw_bolds = sorted(glob(f'{bids_dir}/sub-{sid}/ses-*/func/*_bold.nii.gz')) + \
            sorted(glob(f'{bids_dir}/sub-{sid}/func/*_bold.nii.gz'))
        raw_bolds = filter_(raw_bolds)
        labels = [os.path.basename(_).split(f'sub-{sid}_', 1)[1].rsplit('_bold.nii.gz', 1)[0] for _ in raw_bolds]
        for label in labels:
            label2 = label.replace('-', '_')
            wf_dir = os.path.join(self.work_out, f'func_preproc_{label2}_wf')

            try:
                if os.path.exists(wf_dir):
                    shutil.rmtree(wf_dir)
            except Exception as e:
                print(e)
                return False, str(e)

        fns = []
        for suffix in [
            '_space-MNI152NLin2009cAsym_res-1_desc-preproc_bold.nii.gz',
            '_space-fsaverage5_hemi-R_bold.func.gii',
            '_space-fsaverage5_hemi-L_bold.func.gii',
        ]:
            fns += sorted(glob(f'{self.fmriprep_out}/func/*{suffix}'))
            fns += sorted(glob(f'{self.fmriprep_out}/ses-*/func/*{suffix}'))

        fns = filter_(fns)
        for fn in fns:
            print(fn)
            os.remove(fn)

        return True, ''

    def fmriprep(self, log_name=None, **kwargs):
        if 'anat_only' in kwargs and kwargs['anat_only'] and log_name is None:
            log_name = 'anatonly'
        return self._run_method('fmriprep', log_name=log_name, **kwargs)

    def xform(self, **kwargs):
        return self._run_method('xform', **kwargs)

    def resample(self, filter_=None, log_name=None, **kwargs):
        return self._run_method('resample', filter_=filter_, log_name=log_name, **kwargs)

    def check(self, filter_=None, **kwargs):
        """Check the outputs of one-step surface resampling (fMRIPrep 25+).

        Validate saved time series and coverage, then compare sampling with
        and without susceptibility distortion correction (SDC) against
        fMRIPrep's reference. Runs are discovered once and shared between
        the two checks. Warnings alone do not fail the step.

        Parameters
        ----------
        filter_ : callable or None, optional
            Receives the sorted list of raw BOLD paths once and returns
            the list to check. Default is None, selecting all discovered
            runs. Use the same selection as for :meth:`resample`.
        **kwargs
            Optional arguments for selection, file locations, and diagnostics:

            - ``no_fieldmap`` and ``fieldmap_prefix`` are passed to
              :func:`process.fmriprep.find_runs`, with defaults of an empty
              collection and None. Use the same overrides as for resampling.
            - ``bids_dir``, ``fmriprep_dir``, ``out_dir``, ``xform_dir``,
              ``coverage_dir``, ``combinations``, and ``fs_dir`` override
              this workflow's paths and space/tag pairs, as in resampling.
              Coverage defaults to ``<confound_dir>/coverage``. ``fs_dir``
              is the individual subject's FreeSurfer directory. ``sid``
              is also accepted and should match this workflow's subject;
              report identity and destination always use ``self.sid``.
            - ``n_frames`` is the maximum number of evenly spaced volumes
              for each SDC comparison, default 40. It is expected to be a
              positive integer. See :func:`process.resample.check_sdc` for
              sampling details and memory use.
            - ``low_signal`` and ``low_coverage`` set output-warning
              thresholds, default 0.05 and 0.5. They are passed unchanged
              to :func:`process.resample.check_outputs`.
            - ``n_jobs`` and ``n_depths`` are accepted and ignored so
              resampling settings can be reused. Checking is sequential;
              saved arrays do not encode the sampling depth count.

            Other keywords, including ``log_name`` and ``runs``, are not
            accepted. No finish, running, or error log controls this step.

        Returns
        -------
        bool
            True when the report has no errors. The report itself is saved
            to ``<output_data_root>/checks/sub-<self.sid>.json`` using the
            configured output-data root, even when file paths are overridden.
            Warnings are printed before returning.

        Raises
        ------
        SystemExit
            With status 1 if the report contains errors. The JSON is written
            first, then warnings and errors are printed.
        TypeError
            If an unsupported keyword is supplied; no report is written.
        ValueError
            If derivative versions conflict or differ from the configured
            major/minor version. See :func:`process.fmriprep.check_fmriprep_version`.
        NotImplementedError
            If the resolved fMRIPrep version is not numeric 25 or later.

        Notes
        -----
        Every call reruns the checks and replaces the subject's JSON report,
        including after a previous failure. Its top-level keys are ``sid``,
        ``runs``, ``errors``, and ``warnings``. Each run has the frame count
        and output summaries from ``check_outputs``, plus an ``sdc`` dict
        containing the diagnostic scores and one of these statuses:

        - ``ok``: all scores are finite and ``r_sdc > r_nosdc``;
        - ``not_improved``: all scores are finite and ``r_sdc <= r_nosdc``
          (a warning, not proof of incorrect distortion correction);
        - ``inconclusive``: any score is nonfinite or the diagnostic raises
          (an error). Nonfinite scores become JSON null; a failed diagnostic
          has only its status, with the exception described in ``errors``;
        - ``skipped``: no fieldmap is available (a warning). Discovery allows
          this for acknowledged ``no_fieldmap`` runs and subjects with no
          fieldmaps at all; other missing-fieldmap cases fail discovery.

        A discovery failure is recorded with an empty ``runs`` mapping and
        prevents both checks. Once discovery succeeds, reported file errors
        do not prevent SDC checks, and an SDC failure does not prevent checks
        of later runs. JSON uses ``allow_nan=False``. Unexpected exceptions
        and report-writing failures propagate; a report is not guaranteed
        in those cases or for version and unsupported-keyword errors.

        See Also
        --------
        process.resample.check_outputs : Validate saved arrays and summarize signal and coverage.
        process.resample.check_sdc : Compute correlations and signed fieldmap shift percentiles.
        """
        version = check_fmriprep_version(self.config['fmriprep_out'], self.config['fmriprep_version'])
        major = version.split('.')[0]
        if not (major.isdigit() and int(major) >= 25):
            raise NotImplementedError(f'Checks need outputs of fMRIPrep 25 or later, not {version}.')
        report = self._check(filter_, **kwargs)
        fn = os.path.join(self.config['output_data_root'], 'checks', f'sub-{self.sid}.json')
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, 'w') as f:
            json.dump(report, f, indent=2, allow_nan=False)
        for message in report['warnings']:
            print(f'Warning: {message}')
        if report['errors']:
            for message in report['errors']:
                print(f'Error: {message}')
            exit(1)
        return True

    def compress(self):
        return self._run_method('compress')

    def cleanup(self):
        return self._run_method('cleanup')

    def partial_cleanup(self, filter_, log_name):
        return self._run_method('partial_cleanup', filter_=filter_, log_name=log_name)

    def confound(self, filter_=None, log_name=None):
        return self._run_method('confound', filter_=filter_, log_name=log_name)

    def events(self, filter_=None, log_name=None):
        return self._run_method('events', filter_=filter_, log_name=log_name)

    def archive(self, filter_=None, log_name=None):
        return self._run_method('archive', filter_=filter_, log_name=log_name)

    def anatomy(self):
        return self._run_method('anatomy')

    def t2star(self):
        return self._run_method('t2star')

    def unpack(self, filter_=None):
        pairs = [
            [self.fmriprep_fn, os.path.dirname(self.fmriprep_out)],
            [self.freesurfer_fn, os.path.dirname(self.freesurfer_out)],
        ]

        wf_root = self.work_out
        sid = self.sid
        bids_dir = self.config['bids_dir']
        raw_bolds = sorted(glob(f'{bids_dir}/sub-{sid}/ses-*/func/*_bold.nii.gz')) + \
            sorted(glob(f'{bids_dir}/sub-{sid}/func/*_bold.nii.gz'))
        raw_bolds = raw_bolds if filter_ is None else filter_(raw_bolds)
        out_dir = os.path.join(self.config['output_root'], 'fp_work')
        for raw in raw_bolds:
            label = os.path.basename(raw).split(f'sub-{sid}_', 1)[1].rsplit('_bold.nii.gz', 1)[0]
            lzma_fn = os.path.join(out_dir, f'sub-{sid}_{label}.tar.lzma')
            pairs.append([lzma_fn, wf_root])
        pairs.append([os.path.join(out_dir, f'sub-{sid}_shared.tar.lzma'), wf_root])

        # for lzma_fn, root in pairs:
        #     os.makedirs(root, exist_ok=True)
        #     t0 = datetime.now()
        #     with tarfile.open(lzma_fn) as tf:
        #         tf.extractall(root)
        #     t1 = datetime.now()
        #     print(t1, t1 - t0, lzma_fn, os.path.getsize(lzma_fn))

        jobs = [delayed(unpack_lzma_file)(lzma_fn, root)
            for lzma_fn, root in pairs]
        with Parallel(n_jobs=-1) as parallel:
            parallel(jobs)

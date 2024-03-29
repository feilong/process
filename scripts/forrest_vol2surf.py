import os
from glob import glob
import subprocess
from joblib import Parallel, delayed


if __name__ == '__main__':
    dset, fmriprep_version = 'forrest', '20.2.7'
    bids_dir = os.path.realpath(os.path.expanduser(f'~/lab/BIDS/ds000113'))
    sids = ['01', '02', '03', '04', '05', '06', '09', '10', '14', '15', '16', '17', '18', '19', '20']

    jobs = []
    for sid in sids:
        sd = os.path.expanduser(f'~/lab/fmriprep_out_root/forrest_20.2.7/output_{sid}/freesurfer')
        onavg = os.path.expanduser('~/lab/freesurfer/subjects/onavg-ico32')
        link = os.path.join(sd, 'onavg-ico32')
        if not os.path.exists(link):
            os.symlink(onavg, link)

        wf_root = os.path.join(
            os.path.expanduser(f'~/lab/fmriprep_work_root/{dset}_{fmriprep_version}/work_{sid}'),
            f'fmriprep_wf', f'single_subject_{sid}_wf')

        raw_bolds = sorted(glob(f'{bids_dir}/sub-{sid}/ses-movie/func/*_bold.nii.gz')) + \
            sorted(glob(f'{bids_dir}/sub-{sid}/ses-localizer/func/*_bold.nii.gz'))

        for raw_bold in raw_bolds:
            label = os.path.basename(raw_bold).split(f'sub-{sid}_', 1)[1].rsplit('_bold.nii.gz', 1)[0]
            label2 = label.replace('-', '_')
            wf_dir = (f'{wf_root}/func_preproc_{label2}_wf')

            nii_fn = f'{wf_dir}/bold_t1_trans_wf/merge/vol0000_xform-00000_merged.nii'

            for space in ['onavg-ico32', 'fsavg-ico32']:
                target = 'fsaverage5' if space == 'fsavg-ico32' else space
                for lr in 'lr':
                    out_dir = os.path.join(
                        os.path.expanduser('~/lab/nb-data/forrest/20.2.7/resampled'),
                        space, f'{lr}-cerebrum', '2step_freesurfer7.2')
                    out_fn = os.path.join(out_dir, f'sub-{sid}_{label}.gii')
                    if os.path.exists(out_fn):
                        continue
                    os.makedirs(out_dir, exist_ok=True)

                    lta_fn = f'{wf_dir}/bold_surf_wf/itk2lta/out.lta'
                    if not os.path.exists(lta_fn):
                        os.makedirs(os.path.dirname(lta_fn), exist_ok=True)
                        import nitransforms as nt
                        lta = f'{wf_dir}/../anat_preproc_wf/surface_recon_wf/t1w2fsnative_xfm/out.lta'
                        nt.linear.load(lta, fmt='fs', reference=nii_fn).to_filename(
                            lta_fn, moving=f'{sd}/sub-{sid}/mri/T1.mgz', fmt='fs')

                    cmd = [
                        'mri_vol2surf',
                        # '--cortex',
                        '--hemi', f'{lr}h',
                        '--interp', 'trilinear',
                        '--o', out_fn,
                        '--srcsubject', f'sub-{sid}',
                        '--reg', lta_fn,
                        '--projfrac-avg', '0.000', '1.000', '0.200',
                        '--mov', nii_fn,
                        '--trgsubject', target,
                        '--sd', sd,
                    ]

                    jobs.append(delayed(subprocess.run)(cmd))

                for lr in 'lr':
                    out_dir = os.path.join(
                        os.path.expanduser('~/lab/nb-data/forrest/20.2.7/resampled'),
                        space, f'{lr}-cerebrum', '2step_freesurfer6')
                    out_fn = os.path.join(out_dir, f'sub-{sid}_{label}.gii')
                    if os.path.exists(out_fn):
                        continue
                    os.makedirs(out_dir, exist_ok=True)

                    lta_fn = f'{wf_dir}/bold_surf_wf/itk2lta/out.lta'

                    cmd = [
                        'FS_LICENSE=$HOME/FS_license.txt',
                        'mri_vol2surf',
                        # '--cortex',
                        '--hemi', f'{lr}h',
                        '--interp', 'trilinear',
                        '--o', out_fn,
                        '--srcsubject', f'sub-{sid}',
                        '--reg', lta_fn,
                        '--projfrac-avg', '0.000', '1.000', '0.200',
                        '--mov', nii_fn,
                        '--trgsubject', target,
                        '--sd', sd,
                    ]
                    cmd = ' '.join(cmd)
                    cmd = [
                        'singularity', 'exec', '-e',
                        '-B', '/dartfs:/dartfs',
                        '-B', '/scratch:/scratch',
                        '-B', '/dartfs-hpc:/dartfs-hpc',
                        '-H', os.path.realpath(os.path.expanduser(f'~/lab/singularity_home/fmriprep')),
                        os.path.expanduser('~/lab/fmriprep_20.2.7.sif'),
                        '/bin/bash', '-c', cmd,
                    ]
                    jobs.append(delayed(subprocess.run)(cmd))

    with Parallel(n_jobs=-1) as parallel:
        parallel(jobs)

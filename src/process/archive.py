import os
from glob import glob
from shutil import copy2

from process.compression import copy_files_to_lzma_tar


def archive_subject_work_dir(sid, labels, wf_root, out_dir):
    labels = [_ for _ in labels if '_echo-' not in _ or '_echo-1' in _]
    for label in labels:
        lzma_fn = os.path.join(out_dir, f'sub-{sid}_{label}.tar.lzma')
        label2 = label.replace('-', '_')
        wf_dir = (f'{wf_root}/func_preproc_{label2}_wf')
        todos = []

        xform_dir = os.path.join(
            wf_dir, 'bold_std_trans_wf', '_std_target_MNI152NLin2009cAsym.res1', 'bold_to_std_transform')
        command_fn = os.path.join(xform_dir, 'command.txt')

        with open(command_fn, 'r') as f:
            lines = f.read().splitlines()
        lines = [_ for _ in lines if _ and _ != '-------']
        for cmd in lines:
            parts = cmd.split()
            assert parts[0] == 'antsApplyTransforms'
            assert parts[1] == '--default-value'
            assert parts[2] == '0'
            assert parts[3] == '--float'
            assert parts[4] == '1'
            assert parts[5] == '--input'
            assert parts[6].endswith('.nii.gz') or parts[6].endswith('.nii')
            todos.append(parts[6])

            assert parts[7] == '--interpolation'
            assert parts[8] == 'LanczosWindowedSinc'
            assert parts[9] == '--output'
            assert parts[10].endswith('.nii.gz') or parts[10].endswith('.nii')
            assert parts[11] == '--reference-image'
            assert parts[12] == '/dartfs/rc/lab/H/HaxbyLab/feilong/singularity_home/fmriprep/.cache/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz'
            assert parts[13] == '--transform'
            assert parts[14] == os.path.realpath(os.path.join(wf_root, 'anat_preproc_wf', 'anat_norm_wf', '_template_MNI152NLin2009cAsym', 'registration', 'ants_t1_to_mniComposite.h5'))

            assert parts[15] == '--transform'
            assert parts[16] == os.path.realpath(os.path.join(wf_dir, 'bold_reg_wf', 'bbreg_wf', 'concat_xfm', 'out_fwd.tfm'))
            todos.append(parts[16])

            assert parts[17] == '--transform'
            targets = [
                os.path.realpath(os.path.join(wf_dir, 'sdc_estimate_wf', 'syn_sdc_wf', 'syn', 'ants_susceptibility0Warp.nii.gz')),
                os.path.realpath(os.path.join(wf_dir, 'sdc_estimate_wf', 'pepolar_unwarp_wf', 'cphdr_warp', '_warpfieldQwarp_PLUS_WARP_fixhdr.nii.gz')),
            ] + [os.path.realpath(_) for _ in
                sorted(glob(os.path.join(wf_dir, 'sdc_estimate_wf', 'fmap2field_wf', 'vsm2dfm', '*_phasediff_rads_unwrapped_recentered_filt_demean_maths_fmap_trans_rad_vsm_unmasked_desc-field_sdcwarp.nii.gz')))]
            if parts[18] != 'identity':
                if parts[18] not in targets:
                    print(parts[18])
                    print(targets)
                assert parts[18] in targets
                todos.append(parts[18])
            else:
                todos += [_ for _ in targets if os.path.exists(_)]

            assert parts[19] == '--transform'
            assert len(parts) == 21

        hmc_fn = os.path.join(wf_dir, 'bold_hmc_wf', 'fsl2itk', 'mat2itk.txt')
        todos.append(hmc_fn)
        fn1 = os.path.join(wf_dir, 'bold_t1_trans_wf', 'merge', 'vol0000_xform-00000_clipped_merged.nii')
        fn2 = os.path.join(wf_dir, 'bold_t1_trans_wf', 'merge', 'vol0000_xform-00000_merged.nii')
        if os.path.exists(fn1):
            todos.append(fn1)
        elif os.path.exists(fn2):
            todos.append(fn2)
        else:
            raise Exception(f'Neither {fn1} nor {fn2} exists.')

        copy_files_to_lzma_tar(lzma_fn, todos, rename_func=lambda x: os.path.relpath(x, wf_root))

        shared_fn = os.path.join(out_dir, f'sub-{sid}_shared.tar.lzma')
        lta_fn = os.path.join(wf_root, 'anat_preproc_wf', 'surface_recon_wf',
                't1w2fsnative_xfm', 'out.lta')
        shared = [parts[14], lta_fn]
        if not os.path.exists(shared_fn):
            copy_files_to_lzma_tar(shared_fn, shared, rename_func=lambda x: os.path.relpath(x, wf_root))

import os
from glob import glob
from functools import partial
import numpy as np
import scipy.sparse as sparse
import nibabel as nib
import nitransforms as nt
from joblib import Parallel, delayed
from functools import partial
import neuroboros as nb

from .surface import Hemisphere
from .volume import mni_affine, mni_coords, find_truncation_boundaries, canonical_volume_coords, aseg_mapping, extract_data_in_mni
from .resample import parse_combined_hdf5, compute_warp, parse_warp_image, interpolate


def _average_function(X, xform):
    return X.mean(axis=1) @ xform


class Subject(object):
    def __init__(self, sid, fs_dir=None, wf_root=None, mni_hdf5=None, do_surf=True, do_canonical=True, do_mni=True):
        self.sid = sid
        self.fs_dir = fs_dir
        self.wf_root = wf_root
        self.mni_hdf5 = mni_hdf5
        self.do_surf = do_surf
        self.do_canonical = do_canonical
        self.do_mni = do_mni

        if self.do_surf or self.do_canonical:
            self.prepare_lta()

        if self.do_surf:
            self.prepare_surf()

        if self.do_canonical:
            self.prepare_canonical()

        if self.do_mni:
            self.prepare_mni()

    def prepare_lta(self):
        assert self.wf_root is not None
        lta_fn = os.path.join(
            self.wf_root, 'anat_preproc_wf', 'surface_recon_wf',
            't1w2fsnative_xfm', 'out.lta')
        self.lta = nt.io.lta.FSLinearTransform.from_filename(lta_fn).to_ras()

    def prepare_surf(self):
        assert self.fs_dir is not None
        self.hemispheres = {}
        for lr in 'lr':
            hemi = Hemisphere(lr, self.fs_dir)
            self.hemispheres[lr] = hemi

    def prepare_canonical(self):
        self.brainmask = nib.load(
            os.path.join(self.fs_dir, 'mri', 'brainmask.mgz'))
        canonical = canonical_volume_coords(self.brainmask)
        canonical = canonical @ self.lta.T
        self.canonical_coords = canonical

    def prepare_mni(self):
        assert self.mni_hdf5 is not None
        xyz1 = mni_coords.copy()
        affine, warp, warp_affine = parse_combined_hdf5(self.mni_hdf5)
        np.testing.assert_array_equal(warp_affine, mni_affine)
        diff = compute_warp(xyz1, warp, warp_affine, kwargs={'order': 1})
        xyz1[..., :3] += diff
        xyz1 = xyz1 @ affine.T
        self.mni_coords = xyz1

    def get_surface_data(self, lr, proj='pial'):
        hemi = self.hemispheres[lr]
        coords = hemi.get_coordinates(proj) @ self.lta.T
        return coords

    def get_volume_coords(self, use_mni=True):
        if use_mni:
            return self.mni_coords
        else:
            return self.canonical_coords

    def export_canonical(self, out_dir):
        assert self.fs_dir is not None
        img = nib.load(f'{self.fs_dir}/mri/brainmask.mgz')
        canonical = nib.as_closest_canonical(img)
        boundaries = find_truncation_boundaries(np.asarray(canonical.dataobj))
        affine = canonical.affine.copy()
        affine[:, 3] = affine @ np.concatenate([boundaries[:3, 0], np.array([1.])])
        self.canonical_affine = affine

        for key in ['T1', 'T2', 'FLAIR', 'brainmask', 'ribbon']:
            fs_fn = f'{self.fs_dir}/mri/{key}.mgz'
            if not os.path.exists(fs_fn):
                if key in ['T2', 'FLAIR']:
                    continue
                else:
                    raise Exception(f"FreeSurfer {key} volumn not found.")
            out_fn = f'{out_dir}/canonical/average-volume/freesurfer/sub-{self.sid}_{key}.nii.gz'
            if os.path.exists(out_fn):
                continue
            img = nib.load(fs_fn)
            canonical = nib.as_closest_canonical(img)
            data = np.asarray(canonical.dataobj)
            data = data[boundaries[0, 0]:boundaries[0, 1], boundaries[1, 0]:boundaries[1, 1], boundaries[2, 0]:boundaries[2, 1]]
            new_img = nib.Nifti1Image(data, affine, header=canonical.header)
            os.makedirs(os.path.dirname(out_fn), exist_ok=True)
            new_img.to_filename(out_fn)


class FunctionalRun(object):
    # Temporarily removed the prefiltered data from Interpolator
    def __init__(self, wf_dir, multiecho=False):
        self.wf_dir = wf_dir
        self.has_data = False
        self.multiecho = multiecho

    def load_data(self):
        if self.multiecho:
            self.hmc = None
        else:
            self.hmc = nt.io.itk.ITKLinearTransformArray.from_filename(
                f'{self.wf_dir}/bold_hmc_wf/fsl2itk/mat2itk.txt').to_ras()

        self.ref_to_t1 = nt.io.itk.ITKLinearTransform.from_filename(
            f'{self.wf_dir}/bold_reg_wf/bbreg_wf/concat_xfm/out_fwd.tfm').to_ras()

        nii_fns = sorted(glob(f'{self.wf_dir}/bold_split/vol*.nii.gz'))
        if len(nii_fns) == 0:
            nii_fns = sorted(glob(f'{self.wf_dir}/split_opt_comb/vol*.nii.gz'))
        self.nt = len(nii_fns)

        if self.multiecho:
            self.extra = {}
            t2s_dir = os.path.join(self.wf_dir, 'bold_t2smap_wf', 't2smap_node')
            self.extra['S0map-full'] = nib.load(os.path.join(t2s_dir, 'desc-full_S0map.nii.gz'))
            self.extra['T2starmap-full'] = nib.load(os.path.join(t2s_dir, 'desc-full_T2starmap.nii.gz'))
            self.extra['S0map-limited'] = nib.load(os.path.join(t2s_dir, 'S0map.nii.gz'))
            self.extra['T2starmap-limited'] = nib.load(os.path.join(t2s_dir, 'T2starmap.nii.gz'))

        if self.multiecho:
            warp_fns = []
        else:
            warp_fns = sorted(glob(f'{self.wf_dir}/unwarp_wf/resample/vol*_xfm.nii.gz'))
            if len(warp_fns):
                assert self.nt == len(warp_fns)
            else:
                warp_fns = sorted(glob(f'{self.wf_dir}/sdc_estimate_wf/fmap2field_wf/vsm2dfm/*_sdcwarp.nii.gz'))
                if len(warp_fns) == 0:
                    warp_fns = sorted(glob(f'{self.wf_dir}/sdc_estimate_wf/syn_sdc_wf/syn/ants_susceptibility0Warp.nii.gz'))
                if len(warp_fns) == 0:
                    warp_fns = sorted(glob(f'{self.wf_dir}/sdc_estimate_wf/pepolar_unwarp_wf/cphdr_warp/_warpfieldQwarp_PLUS_WARP_fixhdr.nii.gz'))
                if len(warp_fns) == 0:
                    warp_fns = sorted(glob(os.path.join(self.wf_dir, 'sdc_estimate_wf', 'fmap2field_wf', 'vsm2dfm', '*_phasediff_rads_unwrapped_recentered_filt_demean_maths_fmap_trans_rad_vsm_unmasked_desc-field_sdcwarp.nii.gz')))
                if len(warp_fns) == 1:
                    warp_fns = warp_fns * self.nt
                else:
                    print(self.wf_dir)
                    print(warp_fns)
                    raise Exception

        self.nii_data, self.nii_affines = [], []
        for i, nii_fn in enumerate(nii_fns):
            nii = nib.load(nii_fn)
            data = np.asarray(nii.dataobj)
            self.nii_affines.append(nii.affine)
            self.nii_data.append(data)

        if len(warp_fns):
            self.warp_data, self.warp_affines = [], []
            for i, warp_fn in enumerate(warp_fns):
                warp_data, warp_affine = parse_warp_image(warp_fn)
                self.warp_data.append(warp_data)
                self.warp_affines.append(warp_affine)
        else:
            self.warp_data, self.warp_affines = None, None

        fn1 = f'{self.wf_dir}/bold_t1_trans_wf/merge/vol0000_xform-00000_clipped_merged.nii'
        fn2 = f'{self.wf_dir}/bold_t1_trans_wf/merge/vol0000_xform-00000_merged.nii'
        if os.path.exists(fn1):
            nii = nib.load(fn1)
        elif os.path.exists(fn2):
            nii = nib.load(fn2)
        else:
            raise Exception(f'Neither {fn1} nor {fn2} exists.')
        self.nii_t1 = np.asarray(nii.dataobj)
        self.nii_t1 = [self.nii_t1[..., _] for _ in range(self.nii_t1.shape[-1])]
        self.nii_t1_affine = nii.affine
        self.has_data = True

    def interpolate_extra(
            self, key, coords, interp_kwargs={'order': 1}, fill=np.nan, callback=None,
        ):
        assert self.multiecho
        if not self.has_data:
            self.load_data()

        data = np.asarray(self.extra[key].dataobj)
        affine = self.extra[key].affine.copy()
        coords = coords @ self.ref_to_t1.T

        results = interpolate_original_space_single_volume(
            data, affine, coords, None, None, None, interp_kwargs, fill, callback)
        return results


    def interpolate(
            self, coords, onestep=True, interp_kwargs={'order': 1}, fill=np.nan, callback=None,
            combine_funcs=None, n_jobs=1):
        if not self.has_data:
            self.load_data()

        if onestep:
            interps = interpolate_original_space(
                self.nii_data, self.nii_affines, coords,
                self.ref_to_t1, self.hmc, self.warp_data, self.warp_affines,
                interp_kwargs, fill, callback, combine_funcs, n_jobs=n_jobs)
            return interps
        else:
            interps = interpolate_t1_space(
                self.nii_t1, self.nii_t1_affine, coords,
                interp_kwargs, fill, callback, combine_funcs, n_jobs=n_jobs)
            return interps


def _combine_interpolation_results(interps, n_funcs, combine_funcs, stack=True):
    if n_funcs:
        output = []
        for i in range(n_funcs):
            subset = [interp[i] for interp in interps]
            output.append(
                _combine_interpolation_results(subset, 0, combine_funcs[i], stack))
        return output

    func = np.stack if stack else np.concatenate

    if isinstance(interps[0], np.ndarray):
        output = func(interps, axis=0)
    elif isinstance(interps[0], dict):
        keys = list(interps[0])
        output = {key: [] for key in keys}
        for interp in interps:
            for key in keys:
                output[key].append(interp[key])
        for key in keys:
            output[key] = func(output[key], axis=0)
    else:
        raise ValueError
    if combine_funcs is not None:
        output = combine_funcs(output)
    return output


def run_callback(interp, callback):
    if callback is None:
        return interp
    if isinstance(callback, (list, tuple)):
        output = [func(interp) for func in callback]
        return output
    else:
        output = callback(interp)
        return output


def interpolate_original_space_single_volume(
        data, affine, coords, warp, warp_affine, hmc, interp_kwargs, fill, callback):
    cc = coords.copy()
    if warp is not None:
        diff = compute_warp(cc, warp.astype(np.float64), warp_affine)
        cc[..., :3] += diff
    if hmc is None:
        cc = cc @ np.linalg.inv(affine).T
    else:
        cc = cc @ (hmc.T @ np.linalg.inv(affine).T)
    interp = interpolate(data.astype(np.float64), cc, fill=fill, kwargs=interp_kwargs)
    interp = run_callback(interp, callback)
    return interp


def _run_jobs_and_combine(jobs, callback, combine_funcs, n_jobs):
    n_funcs = len(callback) if isinstance(callback, (list, tuple)) else 0
    n_batches = len(jobs) // n_jobs + int(len(jobs) % n_jobs > 0)
    batch_indices = np.array_split(np.arange(len(jobs)), n_batches)
    results = []
    with Parallel(n_jobs=n_jobs, verbose=0) as parallel:
        for idx in batch_indices:
            batch = [jobs[_] for _ in idx]
            res = parallel(batch)
            res = _combine_interpolation_results(res, n_funcs, combine_funcs)
            results.append(res)
    results = _combine_interpolation_results(results, n_funcs, combine_funcs, stack=False)
    return results


def interpolate_original_space(nii_data, nii_affines, coords,
        ref_to_t1, hmc, warp_data=None, warp_affines=None,
        interp_kwargs={'order': 1}, fill=np.nan, callback=None, combine_funcs=None, n_jobs=1):
    coords = coords @ ref_to_t1.T
    jobs = []
    for i, (data, affine) in enumerate(zip(nii_data, nii_affines)):
        if warp_data is None:
            warp, warp_affine = None, None
        else:
            warp, warp_affine = warp_data[i], warp_affines[i]
        job = delayed(interpolate_original_space_single_volume)(
            data, affine, coords, warp, warp_affine, (None if hmc is None else hmc[i]),
            interp_kwargs, fill, callback)
        jobs.append(job)
    results = _run_jobs_and_combine(jobs, callback, combine_funcs, n_jobs)
    return results


def interpolate_t1_space_single_volume(data, cc, fill, interp_kwargs, callback):
    interp = interpolate(data.astype(np.float64), cc, fill=fill, kwargs=interp_kwargs)
    interp = run_callback(interp, callback)
    return interp


def interpolate_t1_space(nii_t1, nii_t1_affine, coords,
        interp_kwargs={'order': 1}, fill=np.nan, callback=None, combine_funcs=None, n_jobs=1):
    cc = coords @ np.linalg.inv(nii_t1_affine.T)
    jobs = [
        delayed(interpolate_t1_space_single_volume)(
            data, cc, fill, interp_kwargs, callback)
        for data in nii_t1]
    results = _run_jobs_and_combine(jobs, callback, combine_funcs, n_jobs)
    return results


def workflow_single_run(label, sid, wf_root, out_dir, combinations, subj,
        tmpl_dir=os.path.expanduser('~/surface_template/lab/final'), n_jobs=1):
    multiecho = ('_echo-' in label)
    label2 = label.replace('-', '_')
    wf_dir = (f'{wf_root}/func_preproc_{label2}_wf')
    if not os.path.exists(wf_dir):
        for e in range(1, 10):
            wf_dir_ = wf_dir.replace('_echo_1_', f'_echo_{e}_')
            if os.path.exists(wf_dir_):
                wf_dir = wf_dir_
                break
    assert os.path.exists(wf_dir)
    func_run = FunctionalRun(wf_dir, multiecho=multiecho)
    if multiecho:
        label = label.replace('_echo-1', '')

    for lr in 'lr':
        todo = []
        interp_methods = set()
        for space, onestep, proj, resample in combinations:
            tag = '_'.join([('1step' if onestep else '2step'), proj, resample])
            out_fn = f'{out_dir}/{space}/{lr}-cerebrum/{tag}/sub-{sid}_{label}.npy'
            if os.path.exists(out_fn):
                continue
            print(out_fn)
            os.makedirs(os.path.dirname(out_fn), exist_ok=True)
            todo.append((onestep, proj, space, resample, out_fn))
            interp_methods.add((onestep, proj))

        for (onestep, proj) in interp_methods:
            print(onestep, proj)
            coords = subj.get_surface_data(lr, proj)

            selected = [_ for _ in todo if _[:2] == (onestep, proj)]
            funcs, combine_funcs, out_fns = [], [], []
            for sel in selected:
                space, resample, out_fn = sel[2:]

                if space == 'native':
                    funcs.append(lambda x: x.mean(axis=1))
                    combine_funcs.append(None)
                    out_fns.append(out_fn)
                    continue

                sphere_coords = nb.geometry('sphere.reg', lr, space, vertices_only=True)
                xform = subj.hemispheres[lr].get_transformation(sphere_coords, space, resample)
                if resample in ['area', 'overlap']:
                    xform = sparse.diags(np.reciprocal(xform.sum(axis=1).A.ravel())) @ xform
                callback = partial(_average_function, xform=xform)
                funcs.append(callback)
                combine_funcs.append(None)
                out_fns.append(out_fn)

            output = func_run.interpolate(
                coords, onestep, interp_kwargs={'order': 1}, fill=np.nan,
                callback=funcs, combine_funcs=combine_funcs, n_jobs=n_jobs)
            for resampled, out_fn in zip(output, out_fns):
                np.save(out_fn, resampled)
                print(resampled.shape, resampled.dtype, out_fn)

            if func_run.multiecho and onestep:
                for extra_key in func_run.extra:
                    output = func_run.interpolate_extra(extra_key, coords, interp_kwargs={'order': 1}, fill=np.nan, callback=funcs)
                    for resampled, out_fn0 in zip(output, out_fns):
                        out_fn = out_fn0[:-4] + f'_{extra_key}.npy'
                        np.save(out_fn, resampled)
                        print(resampled.shape, resampled.dtype, out_fn)

    todo = []
    funcs = []
    combine_funcs = []
    tag = '1step_linear_overlap'
    for mm in [2, 3, 4]:
        space = f'mni-{mm}mm'
        rois = list(aseg_mapping.values())
        out_fns = [f'{out_dir}/{space}/{roi}/{tag}/sub-{sid}_{label}.npy' for roi in rois]
        if all([os.path.exists(_) for _ in out_fns]):
            continue
        for out_fn in out_fns:
            os.makedirs(os.path.dirname(out_fn), exist_ok=True)
        callback = partial(extract_data_in_mni, mm=mm, cortex=True)
        todo.append(mm)
        funcs.append(callback)
        combine_funcs.append(None)
    out_fn = f'{out_dir}/mni-1mm/average-volume/{tag}/sub-{sid}_{label}.nii.gz'
    if not os.path.exists(out_fn):
        todo.append(1)
        funcs.append(lambda x: x)
        combine_funcs.append(lambda x: np.sum(x, axis=0, keepdims=True))
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)

    if todo:
        coords = subj.get_volume_coords(use_mni=True)
        output = func_run.interpolate(
            coords, True, interp_kwargs={'order': 1}, fill=np.nan, callback=funcs,
            combine_funcs=combine_funcs, n_jobs=n_jobs)
        for mm, res in zip(todo, output):
            space = f'mni-{mm}mm'
            if isinstance(res, dict):
                for roi, resampled in res.items():
                    out_fn = f'{out_dir}/{space}/{roi}/{tag}/sub-{sid}_{label}.npy'
                    np.save(out_fn, resampled)
            else:
                out_fn = f'{out_dir}/{space}/average-volume/{tag}/sub-{sid}_{label}.nii.gz'
                img = nib.Nifti1Image(np.squeeze(res) / func_run.nt, affine=mni_affine)
                img.to_filename(out_fn)

        if func_run.multiecho:
            funcs = [func for func, mm in zip(funcs, todo) if mm != 1]
            todo = [_ for _ in todo if _ != 1]
            print(len(funcs), todo)
            if len(funcs):
                for extra_key in func_run.extra:
                    output = func_run.interpolate_extra(extra_key, coords, interp_kwargs={'order': 1}, fill=np.nan, callback=funcs)
                    for mm, res in zip(todo, output):
                        space = f'mni-{mm}mm'
                        for roi, resampled in res.items():
                            out_fn = f'{out_dir}/{space}/{roi}/{tag}/sub-{sid}_{label}_{extra_key}.npy'
                            np.save(out_fn, resampled)

    tag = '1step_linear_overlap'
    space = 'canonical'
    out_fn = f'{out_dir}/{space}/average-volume/{tag}/sub-{sid}_{label}.nii.gz'
    if not os.path.exists(out_fn):
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
        coords = subj.get_volume_coords(use_mni=False)
        output = func_run.interpolate(
            coords, True, interp_kwargs={'order': 1}, fill=np.nan, callback=[lambda x: x],
            combine_funcs=[lambda x: np.sum(x, axis=0, keepdims=True)], n_jobs=n_jobs)
        img = nib.Nifti1Image(np.squeeze(output[0]) / func_run.nt, affine=subj.canonical_affine)
        img.to_filename(out_fn)

    tag = '1step_fmriprep_overlap'
    for mm in [2, 3, 4]:
        space = f'mni-{mm}mm'
        rois = list(aseg_mapping.values())
        out_fns = [f'{out_dir}/{space}/{roi}/{tag}/sub-{sid}_{label}.npy' for roi in rois]
        if all([os.path.exists(_) for _ in out_fns]):
            continue
        for out_fn in out_fns:
            os.makedirs(os.path.dirname(out_fn), exist_ok=True)

        in_fns = sorted(glob(os.path.join(
            wf_dir, 'bold_std_trans_wf', '_std_target_MNI152NLin2009cAsym.res1',
            'bold_to_std_transform', 'vol*_xform-*.nii.gz')))
        if len(in_fns) == 0:
            continue
        output = []
        for in_fn in in_fns:
            d = np.asanyarray(nib.load(in_fn).dataobj)
            output.append(extract_data_in_mni(d, mm=mm, cortex=True))
        output = _combine_interpolation_results(output, 0, None)

        for roi, resampled in output.items():
            out_fn = f'{out_dir}/{space}/{roi}/{tag}/sub-{sid}_{label}.npy'
            np.save(out_fn, resampled)

    out_fn = f'{out_dir}/mni-1mm/average-volume/{tag}/sub-{sid}_{label}.nii.gz'
    if not os.path.exists(out_fn):
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
        in_fns = sorted(glob(os.path.join(
            wf_dir, 'bold_std_trans_wf', '_std_target_MNI152NLin2009cAsym.res1',
            'bold_to_std_transform', 'vol*_xform-*.nii.gz')))
        if len(in_fns) == 0:
            return
        res = dc_sum(in_fns)
        img = nib.Nifti1Image(res / len(in_fns), affine=mni_affine)
        img.to_filename(out_fn)


def dc_sum(in_fns):
    if len(in_fns) in [1, 2]:
        arr = []
        for in_fn in in_fns:
            d = np.asanyarray(nib.load(in_fn).dataobj)
            arr.append(d)
        return np.sum(arr, axis=0)

    n = len(in_fns) // 2
    arr = dc_sum(in_fns[:n]) + dc_sum(in_fns[n:])
    return arr


def resample_workflow(
        sid, bids_dir, fs_dir, wf_root, out_dir, xform_dir,
        combinations=[
            ('onavg-ico32', '1step_pial_area'),
        ],
        filter_=None,
        n_jobs=1,
    ):

    raw_bolds = sorted(glob(f'{bids_dir}/sub-{sid}/ses-*/func/*_bold.nii.gz')) + \
        sorted(glob(f'{bids_dir}/sub-{sid}/func/*_bold.nii.gz'))
    raw_bolds = [_ for _ in raw_bolds if '_echo-' not in _ or 'echo-1_' in _]
    if filter_ is not None:
        raw_bolds = filter_(raw_bolds)
    labels = [os.path.basename(_).split(f'sub-{sid}_', 1)[1].rsplit('_bold.nii.gz', 1)[0] for _ in raw_bolds]

    new_combinations = []
    for a, b in combinations[::-1]:
        b, c = b.split('_', 1)
        c, d = c.rsplit('_', 1)
        b = {'1step': True, '2step': False}[b]
        new_combinations.append([a, b, c, d])
    combinations = new_combinations

    mni_hdf5 = os.path.join(wf_root, 'anat_preproc_wf', 'anat_norm_wf', '_template_MNI152NLin2009cAsym',
                            'registration', 'ants_t1_to_mniComposite.h5')

    subj = Subject(sid, fs_dir=fs_dir, wf_root=wf_root, mni_hdf5=mni_hdf5, do_surf=True, do_canonical=True, do_mni=True)
    subj.export_canonical(out_dir=out_dir)

    for space, onestep, proj, resample in combinations:
        if space == 'native':
            continue
        for lr in 'lr':
            sphere_coords = nb.geometry('sphere.reg', lr, space, vertices_only=True)

            xform_fn = os.path.join(xform_dir, space, f'{sid}_{resample}_{lr}h.npz')
            if not os.path.exists(xform_fn):
                os.makedirs(os.path.dirname(xform_fn), exist_ok=True)
                xform = subj.hemispheres[lr].get_transformation(sphere_coords, space, resample)
                sparse.save_npz(xform_fn, xform)
            else:
                subj.hemispheres[lr].native[f'to_{space}_{resample}'] = sparse.load_npz(xform_fn)

    for label in labels:
        workflow_single_run(label, sid, wf_root, out_dir, combinations, subj, n_jobs=n_jobs)

import os
from glob import glob
import numpy as np
import nibabel as nib
import nitransforms as nt
from joblib import Parallel, delayed

from .surface import Hemisphere
from .volume import mni_affine, mni_coords, canonical_volume_coords, aseg_mapping, extract_data_in_mni
from .resample import parse_combined_hdf5, compute_warp, parse_warp_image, interpolate


class Subject(object):
    def __init__(self, fs_dir=None, wf_root=None, mni_hdf5=None, do_surf=True, do_canonical=True, do_mni=True):
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

    # def get_surface_data(self, lr, sphere_fn, space, proj='pial', resample='area'):
    #     hemi = self.hemispheres[lr]
    #     coords = hemi.get_coordinates(proj) @ self.lta.T
    #     xform = hemi.get_transformation(sphere_fn, space, resample)
    #     callback = lambda x: x.mean(axis=1) @ xform
    #     return coords, callback

    def get_volume_coords(self, use_mni=True):
        if use_mni:
            return self.mni_coords
        else:
            return self.canonical_coords


class FunctionalRun(object):
    # Temporarily removed the prefiltered data from Interpolator
    def __init__(self, wf_dir):
        self.wf_dir = wf_dir
        self.has_data = False

    def load_data(self):
        self.hmc = nt.io.itk.ITKLinearTransformArray.from_filename(
            f'{self.wf_dir}/bold_hmc_wf/fsl2itk/mat2itk.txt').to_ras()
        self.ref_to_t1 = nt.io.itk.ITKLinearTransform.from_filename(
            f'{self.wf_dir}/bold_reg_wf/bbreg_wf/concat_xfm/out_fwd.tfm').to_ras()

        nii_fns = sorted(glob(f'{self.wf_dir}/bold_split/vol*.nii.gz'))
        warp_fns = sorted(glob(f'{self.wf_dir}/unwarp_wf/resample/vol*_xfm.nii.gz'))
        if len(warp_fns):
            assert len(nii_fns) == len(warp_fns)

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

        nii = nib.load(f'{self.wf_dir}/bold_t1_trans_wf/merge/vol0000_xform-00000_clipped_merged.nii')
        self.nii_t1 = np.asarray(nii.dataobj)
        self.nii_t1 = [self.nii_t1[..., _] for _ in range(self.nii_t1.shape[-1])]
        self.nii_t1_affine = nii.affine
        self.has_data = True

    def interpolate(
            self, coords, onestep=True, interp_kwargs={'order': 1}, fill=np.nan, callback=None, n_jobs=1):
        if not self.has_data:
            self.load_data()

        if onestep:
            interps = interpolate_original_space(
                self.nii_data, self.nii_affines, coords,
                self.ref_to_t1, self.hmc, self.warp_data, self.warp_affines,
                interp_kwargs, fill, callback, n_jobs=n_jobs)
            return interps
        else:
            interps = interpolate_t1_space(
                self.nii_t1, self.nii_t1_affine, coords,
                interp_kwargs, fill, callback, n_jobs=n_jobs)
            return interps


def _combine_interpolation_results(interps, n_funcs):
    if n_funcs:
        output = []
        for i in range(n_funcs):
            subset = [interp[i] for interp in interps]
            output.append(
                _combine_interpolation_results(subset, 0))
        return output

    if isinstance(interps[0], np.ndarray):
        interps = np.stack(interps, axis=0)
        return interps
    if isinstance(interps[0], dict):
        keys = list(interps[0])
        output = {key: [] for key in keys}
        for interp in interps:
            for key in keys:
                output[key].append(interp[key])
        for key in keys:
            output[key] = np.stack(output[key], axis=0)
        return output
    raise ValueError


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
    cc = cc @ (hmc.T @ np.linalg.inv(affine).T)
    interp = interpolate(data.astype(np.float64), cc, fill=fill, kwargs=interp_kwargs)
    interp = run_callback(interp, callback)
    return interp


def _run_jobs_and_combine(jobs, callback, n_jobs):
    n_funcs = len(callback) if isinstance(callback, (list, tuple)) else 0
    n_batches = len(jobs) // n_jobs + int(len(jobs) % n_jobs > 0)
    batches = np.array_split(jobs, n_batches)
    results = []
    with Parallel(n_jobs=n_jobs, verbose=1) as parallel:
        for batch in batches:
            res = parallel(batch)
            res = _combine_interpolation_results(res, n_funcs)
            results.append(res)
    results = _combine_interpolation_results(results, n_funcs)
    return results


def interpolate_original_space(nii_data, nii_affines, coords,
        ref_to_t1, hmc, warp_data=None, warp_affines=None,
        interp_kwargs={'order': 1}, fill=np.nan, callback=None, n_jobs=1):
    coords = coords @ ref_to_t1.T
    jobs = []
    for i, (data, affine) in enumerate(zip(nii_data, nii_affines)):
        if warp_data is None:
            warp, warp_affine = None, None
        else:
            warp, warp_affine = warp_data[i], warp_affines[i]
        job = delayed(interpolate_original_space_single_volume)(
            data, affine, coords, warp, warp_affine, hmc[i], interp_kwargs, fill, callback)
        jobs.append(job)
    results = _run_jobs_and_combine(jobs, callback, n_jobs)
    return results


def interpolate_t1_space_single_volume(data, cc, fill, interp_kwargs, callback):
    interp = interpolate(data.astype(np.float64), cc, fill=fill, kwargs=interp_kwargs)
    interp = run_callback(interp, callback)
    return interp


def interpolate_t1_space(nii_t1, nii_t1_affine, coords,
        interp_kwargs={'order': 1}, fill=np.nan, callback=None, n_jobs=1):
    cc = coords @ np.linalg.inv(nii_t1_affine.T)
    jobs = [
        delayed(interpolate_t1_space_single_volume)(
            data, cc, fill, interp_kwargs, callback)
        for data in nii_t1]
    results = _run_jobs_and_combine(jobs, callback, n_jobs)
    return results


def workflow_single_run(label, sid, wf_root, out_dir, combinations, subj,
        tmpl_dir=os.path.expanduser('~/surface_template/lab/final'), n_jobs=1):
    label2 = label.replace('-', '_')
    wf_dir = (f'{wf_root}/func_preproc_{label2}_wf')
    assert os.path.exists(wf_dir)
    func_run = FunctionalRun(wf_dir)

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
            funcs, out_fns = [], []
            for sel in selected:
                space, resample, out_fn = sel[2:]

                a, b = space.split('-')
                if a == 'fsavg':
                    name = 'fsaverage_' + b
                elif a == 'onavg':
                    name = 'on-avg-1031-final_' + b
                else:
                    name = space
                sphere_fn = f'{tmpl_dir}/{name}_{lr}h_sphere.npz'

                xform = subj.hemispheres[lr].get_transformation(sphere_fn, space, resample)
                callback = lambda x: x.mean(axis=1) @ xform
                funcs.append(callback)
                out_fns.append(out_fn)

            output = func_run.interpolate(
                coords, onestep, interp_kwargs={'order': 1}, fill=np.nan,
                callback=funcs, n_jobs=n_jobs)
            for resampled, out_fn in zip(output, out_fns):
                np.save(out_fn, resampled)
                print(resampled.shape, resampled.dtype, out_fn)

    todo = []
    funcs = []
    for mm in [2, 4]:
        space = f'mni-{mm}mm'
        tag = '1step_linear_overlap'
        rois = list(aseg_mapping.values())
        out_fns = [f'{out_dir}/{space}/{roi}/{tag}/sub-{sid}_{label}.npy' for roi in rois]
        if all([os.path.exists(_) for _ in out_fns]):
            continue
        for out_fn in out_fns:
            os.makedirs(os.path.dirname(out_fn), exist_ok=True)
        callback = lambda x: extract_data_in_mni(x, mm=mm, cortex=True)
        todo.append(mm)
        funcs.append(callback)

    if todo:
        coords = subj.get_volume_coords(use_mni=True)
        output = func_run.interpolate(
            coords, True, interp_kwargs={'order': 1}, fill=np.nan, callback=funcs, n_jobs=n_jobs)
        for mm, res in zip(todo, output):
            for roi, resampled in res.items():
                out_fn = f'{out_dir}/{space}/{roi}/{tag}/sub-{sid}_{label}.npy'
                np.save(out_fn, resampled)

    for mm in [2, 4]:
        space = f'mni-{mm}mm'
        tag = '1step_fmriprep_overlap'
        rois = list(aseg_mapping.values())
        out_fns = [f'{out_dir}/{space}/{roi}/{tag}/sub-{sid}_{label}.npy' for roi in rois]
        if all([os.path.exists(_) for _ in out_fns]):
            continue
        for out_fn in out_fns:
            os.makedirs(os.path.dirname(out_fn), exist_ok=True)

        in_fns = sorted(glob(os.path.join(
            wf_dir, 'bold_std_trans_wf', '_std_target_MNI152NLin2009cAsym.res1',
            'bold_to_std_transform', 'vol*_xform-*.nii.gz')))
        output = []
        for in_fn in in_fns:
            d = np.asanyarray(nib.load(in_fn).dataobj)
            output.append(extract_data_in_mni(d, mm=mm, cortex=True))
        output = _combine_interpolation_results(output, 0)

        for roi, resampled in output.items():
            out_fn = f'{out_dir}/{space}/{roi}/{tag}/sub-{sid}_{label}.npy'
            np.save(out_fn, resampled)


def resample_workflow(
        sid, bids_dir, fs_dir, wf_root, out_dir,
        combinations=[
            ('onavg-ico32', '1step_pial_area'),
        ],
        n_jobs=1,
    ):

    raw_bolds = sorted(glob(f'{bids_dir}/sub-{sid}/ses-*/func/*_bold.nii.gz')) + \
        sorted(glob(f'{bids_dir}/sub-{sid}/func/*_bold.nii.gz'))
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

    subj = Subject(fs_dir=fs_dir, wf_root=wf_root, mni_hdf5=mni_hdf5, do_surf=True, do_canonical=True, do_mni=True)

    for label in labels:
        workflow_single_run(label, sid, wf_root, out_dir, combinations, subj, n_jobs=n_jobs)
    # jobs = [
    #     delayed(workflow_single_run)(label, sid, wf_root, out_dir, combinations, subj)
    #     for label in labels]
    # with Parallel(n_jobs=n_jobs) as parallel:
    #     parallel(jobs)

    # brainmask = nib.load(f'{fs_dir}/mri/brainmask.mgz')
    # canonical = nib.as_closest_canonical(brainmask)
    # boundaries = find_truncation_boundaries(np.asarray(canonical.dataobj))
    # for key in ['T1', 'T2', 'brainmask', 'ribbon']:
    #     out_fn = f'{out_dir}/average-volume/sub-{sid}_{key}.npy'
    #     img = nib.load(f'{fs_dir}/mri/{key}.mgz')
    #     canonical = nib.as_closest_canonical(img)
    #     data = np.asarray(canonical.dataobj)
    #     data = data[boundaries[0, 0]:boundaries[0, 1], boundaries[1, 0]:boundaries[1, 1], boundaries[2, 0]:boundaries[2, 1]]
    #     np.save(out_fn, data)

#     out_fn = f'{out_dir}/average-volume/sub-{sid}_{label}.npy'
#     if not os.path.exists(out_fn):
#         if interpolator is None:
#             interpolator = Interpolator(sid, label, fs_dir, wf_dir)
#             interpolator.prepare(orders=[1])
#         vol = np.mean(interpolator.interpolate_volume(), axis=0)
#         os.makedirs(os.path.dirname(out_fn), exist_ok=True)
#         np.save(out_fn, vol)

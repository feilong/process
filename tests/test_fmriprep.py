import json
import os

import pytest

from process.fmriprep import find_runs

SID = '01'


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').close()


def _add_run(fp, bids, name, ses=None, fmapid=None, version='25.2', ext='.nii.gz', outputs=True):
    """Write a raw BOLD file and its minimal outputs; returns the raw BOLD path."""
    sub_ses = os.path.join(f'sub-{SID}', f'ses-{ses}') if ses else f'sub-{SID}'
    base = os.path.join(fp, sub_ses, 'func', name)
    raw = os.path.join(bids, sub_ses, 'func', name + '_bold' + ext)
    _touch(raw)
    _touch(raw.split('_bold.nii')[0] + '_bold.json')
    _touch(raw.split('_bold.nii')[0] + '_events.tsv')
    if not outputs:
        return raw
    # As in fMRIPrep: transforms are named without `echo` and `part`
    xfm_base = '_'.join(_ for _ in base.split('_') if not _.startswith(('echo-', 'part-')))
    _touch(xfm_base + '_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt')
    _touch(xfm_base + '_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt')
    _touch(base + '_desc-hmc_boldref.nii.gz')
    _touch(base + '_desc-confounds_timeseries.tsv')
    if fmapid is not None:
        desc = '_desc-fmap' if version == '25.2' else ''
        _touch(f'{xfm_base}_from-boldref_to-{fmapid}_mode-image{desc}_xfm.txt')
    return raw


def _add_fieldmap(fp, fmapid, ses=None, levels=('coeff',), prefix=None):
    sub_ses = os.path.join(f'sub-{SID}', f'ses-{ses}') if ses else f'sub-{SID}'
    if prefix is None:
        prefix = f'sub-{SID}_ses-{ses}' if ses else f'sub-{SID}'
    fns = []
    for level in levels:
        fn = os.path.join(fp, sub_ses, 'fmap', f'{prefix}_fmapid-{fmapid}_desc-{level}_fieldmap.nii.gz')
        _touch(fn)
        fns.append(fn)
    _touch(os.path.join(fp, sub_ses, 'fmap', f'{prefix}_fmapid-{fmapid}_desc-preproc_fieldmap.nii.gz'))
    return fns


@pytest.fixture
def dirs(tmp_path):
    return str(tmp_path / 'fmriprep'), str(tmp_path / 'bids')


@pytest.mark.parametrize('version', ['25.1', '25.2'])
def test_find_runs_fieldmap(dirs, version):
    fp, bids = dirs
    coeff_fns = _add_fieldmap(fp, 'auto00000', levels=('coeff0', 'coeff1'))
    _add_fieldmap(fp, 'auto00001')  # another fieldmap, not used by this run
    raw = _add_run(fp, bids, f'sub-{SID}_task-x_run-01', fmapid='auto00000', version=version)
    [run] = find_runs(fp, bids, SID)
    base = os.path.join(fp, f'sub-{SID}', 'func', f'sub-{SID}_task-x_run-01')
    desc = '_desc-fmap' if version == '25.2' else ''
    assert run == {
        'label': 'task-x_run-01',
        'bold_fn': raw,
        'hmc_fn': base + '_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt',
        'coreg_fn': base + '_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt',
        'fieldmap': {
            'xfm_fn': f'{base}_from-boldref_to-auto00000_mode-image{desc}_xfm.txt',
            'coeff_fns': coeff_fns,
            'ref_fn': os.path.join(
                fp, f'sub-{SID}', 'fmap', f'sub-{SID}_fmapid-auto00000_desc-preproc_fieldmap.nii.gz'),
        },
    }
    os.remove(run['fieldmap']['ref_fn'])
    assert find_runs(fp, bids, SID)[0]['fieldmap']['ref_fn'] is None


def test_find_runs_sessions(dirs):
    fp, bids = dirs
    coeff1 = _add_fieldmap(fp, 'auto00000', ses='1')
    coeff2 = _add_fieldmap(fp, 'auto00001', ses='2')
    _add_run(fp, bids, f'sub-{SID}_ses-2_task-x_run-1', ses='2', fmapid='auto00001')
    _add_run(fp, bids, f'sub-{SID}_ses-1_task-x_run-2', ses='1', fmapid='auto00000')
    _add_run(fp, bids, f'sub-{SID}_ses-1_task-x_run-1', ses='1', fmapid='auto00000')
    runs = find_runs(fp, bids, SID)
    assert [_['label'] for _ in runs] == ['ses-1_task-x_run-1', 'ses-1_task-x_run-2', 'ses-2_task-x_run-1']
    assert [_['fieldmap']['coeff_fns'] for _ in runs] == [coeff1, coeff1, coeff2]
    assert all(os.path.exists(_['bold_fn']) for _ in runs)


def test_find_runs_no_fieldmaps(dirs):
    fp, bids = dirs
    _add_run(fp, bids, f'sub-{SID}_task-x')
    _add_run(fp, bids, f'sub-{SID}_task-y')
    runs = find_runs(fp, bids, SID)
    assert [_['label'] for _ in runs] == ['task-x', 'task-y']
    assert all(_['fieldmap'] is None for _ in runs)


def test_find_runs_missing_fieldmap_transform(dirs):
    """A subject with fieldmaps: a run without a transform is an error unless listed."""
    fp, bids = dirs
    _add_fieldmap(fp, 'auto00000')
    _add_run(fp, bids, f'sub-{SID}_task-x', fmapid='auto00000')
    _add_run(fp, bids, f'sub-{SID}_task-y')
    with pytest.raises(ValueError, match='task-y has no boldref-to-fieldmap transform'):
        find_runs(fp, bids, SID)
    runs = find_runs(fp, bids, SID, no_fieldmap={'task-y'})
    assert runs[0]['fieldmap'] is not None and runs[1]['fieldmap'] is None


def test_find_runs_bad_no_fieldmap(dirs):
    fp, bids = dirs
    _add_fieldmap(fp, 'auto00000')
    _add_run(fp, bids, f'sub-{SID}_task-x', fmapid='auto00000')
    with pytest.raises(ValueError, match='listed in no_fieldmap, but has a fieldmap'):
        find_runs(fp, bids, SID, no_fieldmap=['task-x'])
    with pytest.raises(ValueError, match=r"runs that do not exist: \['task-z'\]"):
        find_runs(fp, bids, SID, no_fieldmap=['task-z'])


def test_find_runs_missing_files(dirs):
    fp, bids = dirs
    with pytest.raises(FileNotFoundError, match='No BOLD runs'):
        find_runs(fp, bids, SID)

    _add_run(fp, bids, f'sub-{SID}_task-x', fmapid='auto00000')
    with pytest.raises(FileNotFoundError, match='No coefficient files for fieldmap auto00000'):
        find_runs(fp, bids, SID)
    _add_fieldmap(fp, 'auto00000')
    find_runs(fp, bids, SID)

    base = os.path.join(fp, f'sub-{SID}', 'func', f'sub-{SID}_task-x')
    for suffix in ['_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt',
                   '_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt']:
        os.remove(base + suffix)
        with pytest.raises(FileNotFoundError, match=f'fMRIPrep output .*{suffix} of .* does not exist'):
            find_runs(fp, bids, SID)


def test_find_runs_several_fieldmap_transforms(dirs):
    fp, bids = dirs
    _add_fieldmap(fp, 'auto00000')
    _add_fieldmap(fp, 'auto00001')
    _add_run(fp, bids, f'sub-{SID}_task-x', fmapid='auto00000')
    base = os.path.join(fp, f'sub-{SID}', 'func', f'sub-{SID}_task-x')
    _touch(base + '_from-boldref_to-auto00001_mode-image_desc-fmap_xfm.txt')
    with pytest.raises(ValueError, match='Several boldref-to-fieldmap transforms'):
        find_runs(fp, bids, SID)



def test_find_runs_filter(dirs):
    """A run that fMRIPrep did not process is an error, unless filtered out."""
    fp, bids = dirs
    _add_run(fp, bids, f'sub-{SID}_task-x')
    _add_run(fp, bids, f'sub-{SID}_task-y', outputs=False)
    with pytest.raises(FileNotFoundError, match='task-y_from-orig_to-boldref.* does not exist'):
        find_runs(fp, bids, SID)
    keep_x = lambda fns: [_ for _ in fns if 'task-x' in _]  # noqa: E731
    assert [_['label'] for _ in find_runs(fp, bids, SID, filter_=keep_x)] == ['task-x']
    with pytest.raises(FileNotFoundError, match='No BOLD runs'):
        find_runs(fp, bids, SID, filter_=lambda fns: [])


def test_find_runs_no_fieldmap_filtered(dirs):
    """no_fieldmap labels are checked against all runs, so filtering them out is fine."""
    fp, bids = dirs
    _add_fieldmap(fp, 'auto00000')
    _add_run(fp, bids, f'sub-{SID}_task-x', fmapid='auto00000')
    _add_run(fp, bids, f'sub-{SID}_task-y')
    keep_x = lambda fns: [_ for _ in fns if 'task-x' in _]  # noqa: E731
    runs = find_runs(fp, bids, SID, filter_=keep_x, no_fieldmap=['task-y'])
    assert [_['label'] for _ in runs] == ['task-x']


def test_find_runs_uncompressed(dirs):
    fp, bids = dirs
    raw = _add_run(fp, bids, f'sub-{SID}_task-x', ext='.nii')
    [run] = find_runs(fp, bids, SID)
    assert run['bold_fn'] == raw and run['label'] == 'task-x'


def test_find_runs_multiecho(dirs):
    fp, bids = dirs
    for echo in [1, 2]:
        _add_run(fp, bids, f'sub-{SID}_task-x_echo-{echo}', outputs=False)
    with pytest.raises(ValueError, match='multi-echo, which is not supported'):
        find_runs(fp, bids, SID)


def _fmap_xfm(fp, ses, name, fmapid='auto00000'):
    return os.path.join(fp, f'sub-{SID}', f'ses-{ses}', 'func',
                        f'sub-{SID}_{name}_from-boldref_to-{fmapid}_mode-image_desc-fmap_xfm.txt')


def _write_sources(xfm_fn, sources):
    with open(xfm_fn[:-len('.txt')] + '.json', 'w') as f:
        json.dump({'Sources': sources}, f)


@pytest.fixture
def repeated_ids(dirs):
    """Sessions processed separately: both reuse auto00000."""
    fp, bids = dirs
    coeff1 = _add_fieldmap(fp, 'auto00000', ses='1', levels=('coeff0', 'coeff1'))
    coeff2 = _add_fieldmap(fp, 'auto00000', ses='2', levels=('coeff0', 'coeff1'))
    _add_run(fp, bids, f'sub-{SID}_ses-1_task-x', ses='1', fmapid='auto00000')
    _add_run(fp, bids, f'sub-{SID}_ses-2_task-x', ses='2', fmapid='auto00000')
    return fp, bids, coeff1, coeff2


def test_find_runs_repeated_ids_ambiguous(repeated_ids):
    fp, bids, _, _ = repeated_ids
    with pytest.raises(ValueError, match=r"Several sets .* auto00000 of ses-1_task-x .*fieldmap_prefix=\{'ses-1_task-x'"):
        find_runs(fp, bids, SID)


def test_find_runs_repeated_ids_provenance(repeated_ids):
    """The transform's Sources (fMRIPrep 25.2) name the fieldmap reference."""
    fp, bids, coeff1, coeff2 = repeated_ids
    _write_sources(_fmap_xfm(fp, '1', 'ses-1_task-x'), [
        f'bids::sub-{SID}/ses-1/func/sub-{SID}_ses-1_task-x_desc-hmc_boldref.nii.gz',
        f'bids::sub-{SID}/ses-1/fmap/sub-{SID}_ses-1_fmapid-auto00000_desc-epi_fieldmap.nii.gz'])
    _write_sources(_fmap_xfm(fp, '2', 'ses-2_task-x'), [  # a path inside fMRIPrep's output also works
        f'{fp}/sub-{SID}/ses-2/func/sub-{SID}_ses-2_task-x_desc-hmc_boldref.nii.gz',
        f'{fp}/sub-{SID}/ses-2/fmap/sub-{SID}_ses-2_fmapid-auto00000_desc-epi_fieldmap.nii.gz'])
    assert [_['fieldmap']['coeff_fns'] for _ in find_runs(fp, bids, SID)] == [coeff1, coeff2]


def test_find_runs_repeated_ids_provenance_unhelpful(repeated_ids):
    """Sources that do not point into fMRIPrep's output (e.g., its working directory) are not used."""
    fp, bids, _, _ = repeated_ids
    for ses in '12':
        _write_sources(_fmap_xfm(fp, ses, f'ses-{ses}_task-x'), [
            f'bids::sub-{SID}/ses-{ses}/func/sub-{SID}_ses-{ses}_task-x_desc-hmc_boldref.nii.gz',
            '/work/fmriprep_25_2_wf/fmap_wf/fmap_ref.nii.gz'])
    with pytest.raises(ValueError, match='provenance .* does not identify one'):
        find_runs(fp, bids, SID)


def test_find_runs_repeated_ids_prefix(repeated_ids):
    fp, bids, coeff1, coeff2 = repeated_ids
    # Explicit mapping, including a cross-session one
    runs = find_runs(fp, bids, SID, fieldmap_prefix={'ses-1_task-x': f'sub-{SID}_ses-1',
                                                     'ses-2_task-x': f'sub-{SID}_ses-1'})
    assert [_['fieldmap']['coeff_fns'] for _ in runs] == [coeff1, coeff1]
    runs = find_runs(fp, bids, SID, fieldmap_prefix={'ses-1_task-x': f'sub-{SID}_ses-1',
                                                     'ses-2_task-x': f'sub-{SID}_ses-2'})
    assert [_['fieldmap']['coeff_fns'] for _ in runs] == [coeff1, coeff2]


def test_find_runs_bad_fieldmap_prefix(repeated_ids):
    fp, bids, _, _ = repeated_ids
    with pytest.raises(ValueError, match=r"fieldmap_prefix has labels of runs that do not exist: \['task-z'\]"):
        find_runs(fp, bids, SID, fieldmap_prefix={'task-z': f'sub-{SID}_ses-1'})
    with pytest.raises(ValueError, match="fieldmap_prefix 'sub-01_ses-3' of ses-1_task-x matches 0 sets"):
        find_runs(fp, bids, SID, fieldmap_prefix={'ses-1_task-x': f'sub-{SID}_ses-3'})


def test_find_runs_fieldmap_prefix_without_fieldmap(dirs):
    fp, bids = dirs
    _add_run(fp, bids, f'sub-{SID}_task-x')
    with pytest.raises(ValueError, match='task-x is in fieldmap_prefix, but has no boldref-to-fieldmap'):
        find_runs(fp, bids, SID, fieldmap_prefix={'task-x': f'sub-{SID}'})


def test_find_runs_fieldmap_ambiguous(dirs):
    fp, bids = dirs
    # A session-less set and a session-1 set with the same ID both apply to a session-1 run
    _add_fieldmap(fp, 'auto00000')
    _add_fieldmap(fp, 'auto00000', ses='1')
    _add_run(fp, bids, f'sub-{SID}_ses-1_task-x', ses='1', fmapid='auto00000')
    with pytest.raises(ValueError, match='Several sets of coefficient files for fieldmap auto00000'):
        find_runs(fp, bids, SID)


def test_find_runs_fieldmap_ambiguous_prefix(dirs):
    fp, bids = dirs
    _add_fieldmap(fp, 'auto00000')
    _add_fieldmap(fp, 'auto00000', prefix=f'sub-{SID}_acq-b')
    _add_run(fp, bids, f'sub-{SID}_task-x', fmapid='auto00000')
    with pytest.raises(ValueError, match='Several sets of coefficient files'):
        find_runs(fp, bids, SID)


def test_find_runs_fieldmap_from_other_session(dirs):
    """A run can use a fieldmap of another session (B0FieldSource or IntendedFor)."""
    fp, bids = dirs
    coeff = _add_fieldmap(fp, 'auto00000', ses='1')
    _add_run(fp, bids, f'sub-{SID}_ses-1_task-x', ses='1', fmapid='auto00000')
    _add_run(fp, bids, f'sub-{SID}_ses-2_task-x', ses='2', fmapid='auto00000')
    assert [_['fieldmap']['coeff_fns'] for _ in find_runs(fp, bids, SID)] == [coeff, coeff]


def test_find_runs_session_less_fieldmap(dirs):
    """A fieldmap without a session applies to runs of any session."""
    fp, bids = dirs
    coeff = _add_fieldmap(fp, 'auto00000')
    _add_run(fp, bids, f'sub-{SID}_ses-1_task-x', ses='1', fmapid='auto00000')
    _add_run(fp, bids, f'sub-{SID}_ses-2_task-x', ses='2', fmapid='auto00000')
    assert [_['fieldmap']['coeff_fns'] for _ in find_runs(fp, bids, SID)] == [coeff, coeff]


@pytest.mark.parametrize('fmapid', [None, 'auto00000'])
def test_find_runs_part_mag(dirs, fmapid):
    """All transforms, head motion included, are named without `part`."""
    fp, bids = dirs
    coeff = _add_fieldmap(fp, 'auto00000') if fmapid else None
    raw = _add_run(fp, bids, f'sub-{SID}_task-x_run-1_part-mag', fmapid=fmapid)
    [run] = find_runs(fp, bids, SID)
    func = os.path.join(fp, f'sub-{SID}', 'func')
    assert run['label'] == 'task-x_run-1_part-mag'
    assert run['bold_fn'] == raw
    assert run['hmc_fn'] == os.path.join(
        func, f'sub-{SID}_task-x_run-1_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt')
    assert run['coreg_fn'] == os.path.join(
        func, f'sub-{SID}_task-x_run-1_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt')
    if fmapid:
        assert run['fieldmap'] == {
            'xfm_fn': os.path.join(
                func, f'sub-{SID}_task-x_run-1_from-boldref_to-auto00000_mode-image_desc-fmap_xfm.txt'),
            'coeff_fns': coeff,
            'ref_fn': coeff[0].replace('_desc-coeff_', '_desc-preproc_')}
    else:
        assert run['fieldmap'] is None


def test_find_runs_entities(dirs):
    """Transforms keep sub, ses, task, acq, ce, rec, dir, and run."""
    fp, bids = dirs
    raw = _add_run(fp, bids, f'sub-{SID}_ses-1_task-x_acq-a_ce-b_rec-c_dir-AP_run-1_part-mag', ses='1')
    [run] = find_runs(fp, bids, SID)
    assert run['bold_fn'] == raw and run['label'] == 'ses-1_task-x_acq-a_ce-b_rec-c_dir-AP_run-1_part-mag'
    assert os.path.basename(run['hmc_fn']) == (
        f'sub-{SID}_ses-1_task-x_acq-a_ce-b_rec-c_dir-AP_run-1_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt')


def test_find_runs_chunk(dirs):
    fp, bids = dirs
    _add_run(fp, bids, f'sub-{SID}_task-x_chunk-1', outputs=False)
    with pytest.raises(ValueError, match='chunk entity, which is not supported'):
        find_runs(fp, bids, SID)


@pytest.mark.parametrize('names', [
    ['task-x_part-mag', 'task-x_part-phase'],
    ['task-x', 'task-x_part-mag'],
])
def test_find_runs_same_transforms(dirs, names):
    """Runs that fMRIPrep would name the same transforms for are an error, unless filtered."""
    fp, bids = dirs
    for name in names:
        _add_run(fp, bids, f'sub-{SID}_{name}')
    with pytest.raises(ValueError, match='have the same fMRIPrep transforms'):
        find_runs(fp, bids, SID)
    keep_first = lambda fns: [_ for _ in fns if os.path.basename(_) == f'sub-{SID}_{names[0]}_bold.nii.gz']  # noqa: E731
    assert [_['label'] for _ in find_runs(fp, bids, SID, filter_=keep_first)] == [names[0]]


# load_surfaces

import nibabel as nib
import nitransforms as nt
import numpy as np

from process.fmriprep import load_surfaces

FACES = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int32)
# fsnative -> T1w positions, as fMRIPrep's from-T1w_to-fsnative transform stores it
TO_T1W = np.array([
    [0.9998, -0.02, 0.0, 0.5],
    [0.02, 0.9998, 0.0, -1.0],
    [0.0, 0.0, 1.0, 2.0],
    [0.0, 0.0, 0.0, 1.0]])
CRAS = np.array([1.5, -20.25, 12.0])


def _surface_coords(lr):
    rng = np.random.default_rng('lr'.index(lr))
    white = rng.normal(0, 30, (4, 3))
    return white, white + rng.normal(0, 2, (4, 3))


def _write_gifti(fn, coords):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    nib.save(nib.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(coords.astype(np.float32), intent='NIFTI_INTENT_POINTSET'),
        nib.gifti.GiftiDataArray(FACES, intent='NIFTI_INTENT_TRIANGLE')]), fn)


def _add_gifti_surfaces(fp, ses=None, kinds=('white', 'pial')):
    sub_ses, name = (f'sub-{SID}/ses-{ses}', f'sub-{SID}_ses-{ses}') if ses else (f'sub-{SID}', f'sub-{SID}')
    for lr in 'lr':
        for kind, coords in zip(('white', 'pial'), _surface_coords(lr)):
            if kind in kinds:
                _write_gifti(os.path.join(fp, sub_ses, 'anat', f'{name}_hemi-{lr.upper()}_{kind}.surf.gii'), coords)
    # Other surfaces that must not be picked up
    _write_gifti(os.path.join(fp, sub_ses, 'anat', f'{name}_hemi-L_midthickness.surf.gii'), np.zeros((4, 3)))
    _write_gifti(os.path.join(fp, sub_ses, 'anat', f'{name}_hemi-L_space-fsLR_desc-reg_sphere.surf.gii'),
                 np.zeros((4, 3)))


def _add_freesurfer(fp, fs_dir, xfm=True):
    """FreeSurfer surfaces whose T1w-space positions equal `_surface_coords`."""
    os.makedirs(os.path.join(fs_dir, 'surf'))
    info = {'head': np.array([2, 0, 20], dtype=np.int32), 'valid': '1  # volume info valid',
            'filename': 'T1.mgz', 'volume': np.array([256, 256, 256]),
            'voxelsize': np.ones(3), 'xras': np.array([-1.0, 0, 0]), 'yras': np.array([0, 0, -1.0]),
            'zras': np.array([0, 1.0, 0]), 'cras': CRAS}
    to_fs = np.linalg.inv(TO_T1W)
    for lr in 'lr':
        for kind, coords in zip(('white', 'pial'), _surface_coords(lr)):
            tkr = coords @ to_fs[:3, :3].T + to_fs[:3, 3] - CRAS
            nib.freesurfer.write_geometry(
                os.path.join(fs_dir, 'surf', f'{lr}h.{kind}'), tkr, FACES, volume_info=info)
    anat = os.path.join(fp, f'sub-{SID}', 'anat')
    os.makedirs(anat, exist_ok=True)
    if xfm:
        nt.linear.Affine(TO_T1W).to_filename(
            os.path.join(anat, f'sub-{SID}_from-T1w_to-fsnative_mode-image_xfm.txt'), fmt='itk')
        nt.linear.Affine(to_fs).to_filename(
            os.path.join(anat, f'sub-{SID}_from-fsnative_to-T1w_mode-image_xfm.txt'), fmt='itk')


@pytest.mark.parametrize('ses', [None, '1'])
@pytest.mark.parametrize('lr', ['l', 'r'])
def test_load_surfaces(dirs, ses, lr):
    fp, _ = dirs
    _add_gifti_surfaces(fp, ses)
    white, pial = load_surfaces(fp, SID, lr)
    assert white.dtype == pial.dtype == np.float64
    for coords, expected in zip((white, pial), _surface_coords(lr)):
        np.testing.assert_allclose(coords, expected, atol=1e-4)  # GIFTI stores float32


@pytest.mark.parametrize('lr', ['l', 'r'])
def test_load_surfaces_freesurfer_fallback(dirs, tmp_path, lr):
    fp, _ = dirs
    fs_dir = str(tmp_path / 'freesurfer' / f'sub-{SID}')
    _add_freesurfer(fp, fs_dir)
    white, pial = load_surfaces(fp, SID, lr, fs_dir=fs_dir)
    assert white.dtype == pial.dtype == np.float64
    for coords, expected in zip((white, pial), _surface_coords(lr)):
        np.testing.assert_allclose(coords, expected, atol=1e-3)  # float32 and ITK text precision

    # fMRIPrep's surfaces are preferred when both exist
    _add_gifti_surfaces(fp)
    os.remove(os.path.join(fs_dir, 'surf', f'{lr}h.white'))
    load_surfaces(fp, SID, lr, fs_dir=fs_dir)


def test_load_surfaces_errors(dirs, tmp_path):
    fp, _ = dirs
    fs_dir = str(tmp_path / 'freesurfer' / f'sub-{SID}')
    _add_freesurfer(fp, fs_dir, xfm=False)
    with pytest.raises(ValueError, match='lr'):
        load_surfaces(fp, SID, 'L')
    with pytest.raises(FileNotFoundError, match='--level full'):
        load_surfaces(fp, SID, 'l')
    with pytest.raises(FileNotFoundError, match='T1w-to-fsnative'):
        load_surfaces(fp, SID, 'l', fs_dir=fs_dir)

    _add_gifti_surfaces(fp, kinds=('white',))
    with pytest.raises(ValueError, match='without its counterpart'):
        load_surfaces(fp, SID, 'l', fs_dir=fs_dir)
    _add_gifti_surfaces(fp)
    _add_gifti_surfaces(fp, ses='1')
    with pytest.raises(ValueError, match='Several pairs'):
        load_surfaces(fp, SID, 'l')


def test_load_surfaces_no_pairing_across_sessions(dirs):
    """White from one session and pial from another may be in different T1w frames."""
    fp, _ = dirs
    _add_gifti_surfaces(fp, ses='1', kinds=('white',))
    _add_gifti_surfaces(fp, ses='2', kinds=('pial',))
    with pytest.raises(ValueError, match='without its counterpart'):
        load_surfaces(fp, SID, 'l')
    # A complete pair elsewhere does not excuse the incomplete ones
    _add_gifti_surfaces(fp)
    with pytest.raises(ValueError, match='without its counterpart'):
        load_surfaces(fp, SID, 'l')


# fmriprep_version

import json

from process.fmriprep import fmriprep_version


@pytest.mark.parametrize('description, expected', [
    ({'GeneratedBy': [{'Name': 'fMRIPrep', 'Version': '25.2.7'}]}, '25.2.7'),
    ({'GeneratedBy': [{'Name': 'other', 'Version': '1.0'}, {'Name': 'fMRIPrep', 'Version': '25.1.4'}]},
     '25.1.4'),
    ({'PipelineDescription': {'Name': 'fMRIPrep', 'Version': '20.2.7'}}, '20.2.7'),
    ({'GeneratedBy': [{'Name': 'other', 'Version': '1.0'}]}, None),
    ({'GeneratedBy': [{'Name': 'fMRIPrep'}]}, None),
    ({'Name': 'fMRIPrep outputs'}, None),
    (None, None),
])
def test_fmriprep_version(tmp_path, description, expected):
    if description is not None:
        (tmp_path / 'dataset_description.json').write_text(json.dumps(description))
    assert fmriprep_version(str(tmp_path)) == expected


# check_fmriprep_version

from process.fmriprep import check_fmriprep_version


def _describe(path, version):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'dataset_description.json').write_text(
        json.dumps({'GeneratedBy': [{'Name': 'fMRIPrep', 'Version': version}]}))


@pytest.mark.filterwarnings('error')
@pytest.mark.parametrize('nested', [False, True])
def test_check_fmriprep_version(tmp_path, nested):
    _describe(tmp_path / 'fmriprep' if nested else tmp_path, '25.2.7')
    assert check_fmriprep_version(str(tmp_path), '25.2.7') == '25.2.7'


def test_check_fmriprep_version_warnings(tmp_path):
    with pytest.warns(UserWarning, match='Cannot read'):
        assert check_fmriprep_version(str(tmp_path), '20.2.7') == '20.2.7'
    _describe(tmp_path, '25.2.7')
    with pytest.warns(UserWarning, match='differs'):
        assert check_fmriprep_version(str(tmp_path), '25.2.6') == '25.2.7'


@pytest.mark.parametrize('nested, actual, configured', [
    (False, '25.2.7', '20.2.7'),  # 20.x paths would point into <out>/fmriprep
    (False, '24.1.1', '25.2.7'),  # the working directory would be fmriprep_25_2_wf
    (False, '25.1.4', '25.2.7'),
    (True, '20.2.7', '25.2.7'),
    (False, '25.2.7', 'next.1'),
])
def test_check_fmriprep_version_incompatible(tmp_path, nested, actual, configured):
    _describe(tmp_path / 'fmriprep' if nested else tmp_path, actual)
    with pytest.raises(ValueError, match='configured'):
        check_fmriprep_version(str(tmp_path), configured)


def test_check_fmriprep_version_conflict(tmp_path):
    _describe(tmp_path, '25.2.7')
    _describe(tmp_path / 'fmriprep', '20.2.7')
    with pytest.raises(ValueError, match='Several'):
        check_fmriprep_version(str(tmp_path), '25.2.7')

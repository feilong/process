import json
import os

import pytest

from process.resample import find_runs, list_bold_runs

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
        },
    }


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



def test_list_bold_runs(dirs):
    fp, bids = dirs
    a = _add_run(fp, bids, f'sub-{SID}_ses-2_task-x', ses='2')
    b = _add_run(fp, bids, f'sub-{SID}_ses-1_task-y', ses='1', ext='.nii')
    c = _add_run(fp, bids, f'sub-{SID}_task-z')
    _touch(os.path.join(bids, f'sub-{SID}', 'func', f'sub-{SID}_task-z_sbref.nii.gz'))
    _touch(os.path.join(bids, f'sub-{SID}', 'anat', f'sub-{SID}_T1w.nii.gz'))
    _touch(os.path.join(bids, 'sub-011', 'func', 'sub-011_task-x_bold.nii.gz'))  # other subject
    assert list_bold_runs(bids, SID) == sorted([a, b, c])
    assert list_bold_runs(bids, SID, filter_=lambda fns: [_ for _ in fns if 'ses-' in _]) == sorted([a, b])


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
            'coeff_fns': coeff}
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

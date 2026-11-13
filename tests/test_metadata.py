import json
import sys

import numpy as np
import nibabel as nib
import pytest

from process.metadata import (
    parse_bids_filename, get_bids_metadata, get_trt, get_trt_bids,
    distortion_parameters)


NPE = 64  # voxels along the phase-encoding axis (j) of the test image


@pytest.fixture
def epi(tmp_path):
    fn = tmp_path / 'sub-01_task-x_bold.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((70, NPE, 3), dtype=np.float32), np.eye(4)), fn)
    return str(fn)


# Ways of specifying the readout time: (metadata, get_trt kwargs)
CASES = {
    'trt': ({'TotalReadoutTime': 0.05251}, {}),
    'trt-and-ees': (
        {'TotalReadoutTime': 0.05, 'PhaseEncodingDirection': 'j', 'EffectiveEchoSpacing': 0.1}, {}),
    'trt-zero': ({'TotalReadoutTime': 0}, {'fallback': 0.03125}),
    'est-trt': ({'EstimatedTotalReadoutTime': 0.04}, {}),
    'est-trt-used': ({'EstimatedTotalReadoutTime': 0.04}, {'use_estimate': True}),
    'ees': ({'PhaseEncodingDirection': 'j-', 'EffectiveEchoSpacing': 0.00059}, {}),
    'ees-no-pe': ({'EffectiveEchoSpacing': 0.00059}, {}),
    'ees-and-est-trt': (
        {'PhaseEncodingDirection': 'j', 'EffectiveEchoSpacing': 0.00059,
         'EstimatedTotalReadoutTime': 0.04}, {}),
    'ees-and-est-trt-used': (
        {'PhaseEncodingDirection': 'j', 'EffectiveEchoSpacing': 0.00059,
         'EstimatedTotalReadoutTime': 0.04}, {'use_estimate': True}),
    'est-ees-used': (
        {'PhaseEncodingDirection': 'j', 'EstimatedEffectiveEchoSpacing': 0.0006},
        {'use_estimate': True}),
    'echo-spacing': (
        {'PhaseEncodingDirection': 'j', 'EchoSpacing': 0.0011,
         'ParallelReductionFactorInPlane': 3}, {}),
    'philips': (
        {'PhaseEncodingDirection': 'j', 'WaterFatShift': 14.256, 'EPIFactor': 35,
         'ImagingFrequency': 127.7}, {}),
    'philips-estimates-used': (
        {'PhaseEncodingDirection': 'j', 'WaterFatShift': 14.256, 'EPIFactor': 35,
         'ImagingFrequency': 127.7, 'EstimatedTotalReadoutTime': 0.02}, {'use_estimate': True}),
    'pe-only-fallback': ({'PhaseEncodingDirection': 'j'}, {'fallback': 0.03125}),
    'empty': ({}, {}),
}

# Expected results of get_trt_bids; None means ValueError
BIDS_EXPECTED = {
    'trt': 0.05251,
    'trt-and-ees': 0.05,
    'ees': 0.00059 * (NPE - 1),
    'ees-and-est-trt': 0.00059 * (NPE - 1),
}


@pytest.mark.parametrize('name', CASES)
def test_get_trt_bids(name, epi):
    meta, kwargs = CASES[name]
    expected = BIDS_EXPECTED.get(name)
    use_estimate = kwargs.get('use_estimate', False)
    if expected is None:
        with pytest.raises(ValueError):
            get_trt_bids(meta, epi, use_estimate=use_estimate)
    else:
        assert get_trt_bids(meta, epi, use_estimate=use_estimate) == pytest.approx(expected)


@pytest.mark.parametrize('name', CASES)
def test_get_trt_bids_agrees_with_sdcflows(name, epi):
    """get_trt_bids gives sdcflows' value, or raises; never a different value."""
    epimanip = pytest.importorskip('sdcflows.utils.epimanip')
    meta, kwargs = CASES[name]
    try:
        ours = get_trt_bids(meta, epi, use_estimate=kwargs.get('use_estimate', False))
    except ValueError:
        return
    assert ours == epimanip.get_trt(meta, epi, **kwargs)


@pytest.mark.parametrize('name', CASES)
def test_get_trt_uses_sdcflows(name, epi):
    epimanip = pytest.importorskip('sdcflows.utils.epimanip')
    meta, kwargs = CASES[name]
    try:
        expected = epimanip.get_trt(meta, epi, **kwargs)
    except ValueError:
        with pytest.raises(ValueError):
            get_trt(meta, epi, **kwargs)
    else:
        assert get_trt(meta, epi, **kwargs) == expected


@pytest.fixture
def no_sdcflows(monkeypatch):
    """Make `from sdcflows.utils.epimanip import ...` raise ImportError."""
    monkeypatch.setitem(sys.modules, 'sdcflows.utils.epimanip', None)


@pytest.mark.parametrize('name', CASES)
def test_get_trt_without_sdcflows(name, epi, no_sdcflows):
    meta, kwargs = CASES[name]
    expected = BIDS_EXPECTED.get(name)
    if expected is None:
        with pytest.raises(ValueError):
            get_trt(meta, epi, **kwargs)
    else:
        assert get_trt(meta, epi, **kwargs) == pytest.approx(expected)


def test_get_trt_bids_needs_in_file():
    assert get_trt_bids({'TotalReadoutTime': 0.05}) == 0.05
    with pytest.raises(ValueError):
        get_trt_bids({'PhaseEncodingDirection': 'j', 'EffectiveEchoSpacing': 0.0006})


def test_parse_bids_filename():
    assert parse_bids_filename('/a/sub-01_ses-2_task-rest_run-1_bold.nii.gz') == {
        'sub': '01', 'ses': '2', 'task': 'rest', 'run': '1',
        'suffix': 'bold', 'ext': '.nii.gz'}
    assert parse_bids_filename('task-rest_bold.json') == {
        'task': 'rest', 'suffix': 'bold', 'ext': '.json'}


def _write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


@pytest.fixture
def bids(tmp_path):
    root = tmp_path / 'bids'
    func = root / 'sub-01' / 'ses-1' / 'func'
    func.mkdir(parents=True)
    _write_json(root / 'dataset_description.json', {'Name': 'test', 'BIDSVersion': '1.8.0'})
    _write_json(root / 'task-x_bold.json',
                {'TotalReadoutTime': 0.1, 'RepetitionTime': 2.0, 'PhaseEncodingDirection': 'j'})
    _write_json(root / 'task-y_bold.json', {'TotalReadoutTime': 9.0})       # other task
    _write_json(root / 'task-x_sbref.json', {'RepetitionTime': 9.0})        # other suffix
    _write_json(root / 'sub-01' / 'sub-01_task-x_bold.json', {'TotalReadoutTime': 0.2})
    _write_json(root / 'sub-01' / 'ses-1' / 'sub-01_ses-1_task-x_bold.json', {'EchoTime': 0.03})
    _write_json(func / 'sub-01_ses-1_task-x_run-1_bold.json', {'PhaseEncodingDirection': 'j-'})
    _write_json(func / 'sub-01_ses-1_task-x_run-2_bold.json', {'TotalReadoutTime': 9.0})
    bold = func / 'sub-01_ses-1_task-x_run-1_bold.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((10, NPE, 3), dtype=np.float32), np.eye(4)), bold)
    return root, str(bold)


def test_get_bids_metadata(bids):
    root, bold = bids
    expected = {'TotalReadoutTime': 0.2, 'RepetitionTime': 2.0,
                'PhaseEncodingDirection': 'j-', 'EchoTime': 0.03}
    assert get_bids_metadata(bold, root) == expected


def _func_bold(tmp_path, name='sub-01_task-x_run-1_bold.nii.gz'):
    root = tmp_path / 'bids'
    func = root / 'sub-01' / 'func'
    func.mkdir(parents=True)
    bold = func / name
    bold.touch()
    return root, func, bold


def test_get_bids_metadata_same_directory_specificity(tmp_path):
    root, func, bold = _func_bold(tmp_path)
    _write_json(func / 'sub-01_task-x_run-1_bold.json', {'B': 2})
    _write_json(func / 'sub-01_task-x_bold.json', {'A': 1, 'B': 1})
    _write_json(func / 'sub-01_bold.json', {'A': 0, 'B': 0, 'C': 0})
    with pytest.warns(UserWarning, match='most specific'):
        assert get_bids_metadata(str(bold), str(root)) == {'A': 1, 'B': 2, 'C': 0}


@pytest.mark.parametrize('names', [
    ['sub-01_task-x_bold.json', 'sub-01_run-1_bold.json'],           # same number of entities
    ['task-x_bold.json', 'sub-01_run-1_bold.json'],                  # not nested
])
def test_get_bids_metadata_same_directory_ambiguous(tmp_path, names):
    root, func, bold = _func_bold(tmp_path)
    for name in names:
        _write_json(func / name, {'TotalReadoutTime': 0.1})
    with pytest.raises(ValueError, match='none of them is the most specific'):
        get_bids_metadata(str(bold), str(root))


def test_get_bids_metadata_unknown_entity(tmp_path):
    root, func, bold = _func_bold(tmp_path)
    _write_json(func / 'sub-01_task-x_foo-bar_bold.json', {'TotalReadoutTime': 0.1})
    _write_json(func / 'sub-01_task-x_foo-bar_sbref.json', {})  # other suffix, ignored
    with pytest.raises(ValueError, match="not defined by BIDS: \\['foo'\\]"):
        get_bids_metadata(str(bold), str(root))


def test_get_bids_metadata_null(tmp_path):
    """Null values are dropped, including those that override inherited values."""
    root, func, bold = _func_bold(tmp_path)
    _write_json(root / 'task-x_bold.json', {'A': 1, 'B': 1, 'C': None})
    _write_json(func / 'sub-01_task-x_run-1_bold.json', {'A': None, 'B': 2})
    assert get_bids_metadata(str(bold), str(root)) == {'B': 2}


def test_get_bids_metadata_outside_root(tmp_path):
    root, func, bold = _func_bold(tmp_path)
    with pytest.raises(ValueError, match='is not inside'):
        get_bids_metadata(str(bold), str(func / 'sub-01'))
    with pytest.raises(ValueError, match='is not inside'):
        get_bids_metadata(str(bold), str(tmp_path / 'bid'))


def test_get_bids_metadata_zero_padded_index(tmp_path):
    """Index entities match as integers (run-1 is run-01); labels match as strings."""
    root = tmp_path / 'bids'
    func = root / 'sub-01' / 'func'
    _write_json(root / 'task-x_bold.json', {'PhaseEncodingDirection': 'j', 'TotalReadoutTime': 0.1})
    _write_json(func / 'sub-01_task-x_run-1_bold.json',
                {'PhaseEncodingDirection': 'j-', 'TotalReadoutTime': 0.05})
    _write_json(func / 'sub-1_task-x_run-01_bold.json', {'TotalReadoutTime': 9.0})  # other subject
    _write_json(func / 'sub-01_task-x_run-10_bold.json', {'TotalReadoutTime': 9.0})  # other run
    bold = func / 'sub-01_task-x_run-01_bold.nii.gz'
    bold.touch()
    assert get_bids_metadata(str(bold), str(root)) == {
        'PhaseEncodingDirection': 'j-', 'TotalReadoutTime': 0.05}


def test_distortion_parameters(bids):
    root, bold = bids
    assert distortion_parameters(bold, root) == ('j-', 0.2)


@pytest.mark.parametrize('pe_dir', [None, '', 'j--', 'y', 'J', ['j']])
def test_distortion_parameters_invalid_pe(tmp_path, pe_dir):
    root = tmp_path / 'bids'
    func = root / 'sub-01' / 'func'
    meta = {'TotalReadoutTime': 0.05}
    if pe_dir is not None:
        meta['PhaseEncodingDirection'] = pe_dir
    _write_json(func / 'sub-01_task-x_bold.json', meta)
    bold = func / 'sub-01_task-x_bold.nii.gz'
    bold.touch()
    with pytest.raises(ValueError, match='PhaseEncodingDirection'):
        distortion_parameters(str(bold), str(root))


def _estimated_only_bold(tmp_path):
    root = tmp_path / 'bids'
    func = root / 'sub-01' / 'func'
    bold = func / 'sub-01_task-x_bold.nii.gz'
    _write_json(func / 'sub-01_task-x_bold.json',
                {'PhaseEncodingDirection': 'i', 'EstimatedTotalReadoutTime': 0.04})
    nib.save(nib.Nifti1Image(np.zeros((NPE, 10, 3), dtype=np.float32), np.eye(4)), bold)
    return str(bold), str(root)


def test_distortion_parameters_fallback(tmp_path):
    pytest.importorskip('sdcflows.utils.epimanip')
    bold, root = _estimated_only_bold(tmp_path)
    with pytest.raises(ValueError):
        distortion_parameters(bold, root)
    assert distortion_parameters(bold, root, fallback='estimated') == ('i', 0.04)
    assert distortion_parameters(bold, root, fallback=0.03125) == ('i', 0.03125)


def test_distortion_parameters_fallback_without_sdcflows(tmp_path, no_sdcflows):
    bold, root = _estimated_only_bold(tmp_path)
    for fallback in [None, 'estimated', 0.03125]:
        with pytest.raises(ValueError):
            distortion_parameters(bold, root, fallback=fallback)

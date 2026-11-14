import json
import sys

import numpy as np
import nibabel as nib
import pytest

from process.metadata import get_trt, get_trt_bids, distortion_parameters


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


def _write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


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

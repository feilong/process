"""Dispatch of the resample step of PreprocessWorkflow by fMRIPrep version."""
import importlib
import json
import sys
from unittest import mock

import pytest

NEUROBOROS = ['neuroboros', 'neuroboros.surface', 'neuroboros.surface.voronoi',
              'neuroboros.surface.properties', 'neuroboros.surface.nnfr', 'neuroboros.surface.areal']


@pytest.fixture
def main(monkeypatch):
    """`process.main`, with `neuroboros` stubbed if it is not installed."""
    try:
        import neuroboros  # noqa: F401
    except ImportError:
        for name in NEUROBOROS:
            monkeypatch.setitem(sys.modules, name, mock.MagicMock())
    main = importlib.import_module('process.main')
    calls = []
    monkeypatch.setattr(main, 'resample_surfaces', lambda **kwargs: calls.append(('new', kwargs)))
    monkeypatch.setattr(main, 'resample_workflow', lambda **kwargs: calls.append(('old', kwargs)))
    return main, calls


def _workflow(main, tmp_path, configured, actual=None, nested=False):
    out = tmp_path / 'fmriprep_out'
    if actual is not None:
        d = out / 'fmriprep' if nested else out
        d.mkdir(parents=True)
        (d / 'dataset_description.json').write_text(
            json.dumps({'GeneratedBy': [{'Name': 'fMRIPrep', 'Version': actual}]}))
    return main.PreprocessWorkflow({
        'sid': '01', 'fmriprep_version': configured, 'n_procs': 2, 'bids_dir': str(tmp_path / 'bids'),
        'output_root': str(tmp_path / 'archive'), 'output_data_root': str(tmp_path / 'nb-data'),
        'output_summary_root': str(tmp_path / 'summary'), 'fmriprep_out': str(out),
        'fmriprep_work': str(tmp_path / 'work'),
        'combinations': [('onavg-ico32', '1step_pial_overlap')]})


@pytest.mark.filterwarnings('error')
def test_resample_new(main, tmp_path):
    main, calls = main
    wf = _workflow(main, tmp_path, '25.2.7', '25.2.7')
    assert wf._run_resample(filter_=None, no_fieldmap=('task-x',)) == (True, '')
    [(kind, kwargs)] = calls
    out = tmp_path / 'fmriprep_out'
    assert kind == 'new'
    assert kwargs == dict(
        sid='01', bids_dir=str(tmp_path / 'bids'), fmriprep_dir=str(out),
        out_dir=str(tmp_path / 'nb-data' / 'resampled'), xform_dir=str(tmp_path / 'nb-data' / 'xforms'),
        coverage_dir=str(tmp_path / 'nb-data' / 'confounds' / 'coverage'),
        combinations=[('onavg-ico32', '1step_pial_overlap')],
        fs_dir=str(out / 'sourcedata' / 'freesurfer' / 'sub-01'),
        filter_=None, n_jobs=2, no_fieldmap=('task-x',))


@pytest.mark.filterwarnings('error')
@pytest.mark.parametrize('version, nested, work', [
    ('20.2.7', True, 'fmriprep_wf'),
    ('24.1.1', False, 'fmriprep_24_1_wf'),
])
def test_resample_old(main, tmp_path, version, nested, work):
    main, calls = main
    wf = _workflow(main, tmp_path, version, version, nested=nested)
    assert wf._run_resample(filter_=None) == (True, '')
    [(kind, kwargs)] = calls
    assert kind == 'old'
    assert kwargs['wf_root'] == str(tmp_path / 'work' / work / 'single_subject_01_wf')


def test_resample_patch_mismatch_and_unreadable(main, tmp_path):
    main, calls = main
    with pytest.warns(UserWarning, match='differs'):
        _workflow(main, tmp_path / 'a', '25.2.6', '25.2.7')._run_resample(filter_=None)
    with pytest.warns(UserWarning, match='Cannot read'):
        _workflow(main, tmp_path / 'b', '25.2.7')._run_resample(filter_=None)
    with pytest.warns(UserWarning, match='Cannot read'):
        _workflow(main, tmp_path / 'c', '20.2.7')._run_resample(filter_=None)
    assert [kind for kind, _ in calls] == ['new', 'new', 'old']


@pytest.mark.parametrize('configured, actual, nested', [
    ('20.2.7', '25.2.7', False),
    ('25.2.7', '24.1.1', False),
    ('25.2.7', '20.2.7', True),
])
def test_resample_incompatible_versions(main, tmp_path, configured, actual, nested):
    main, calls = main
    wf = _workflow(main, tmp_path, configured, actual, nested=nested)
    with pytest.raises(ValueError, match='configured'):
        wf._run_resample(filter_=None)
    assert not calls

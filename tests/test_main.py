"""The resample and check steps of PreprocessWorkflow."""
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


SDC = {'r_sdc': 0.9, 'r_nosdc': 0.8, 'shift_p1': -2.0, 'shift_p99': 3.0}


@pytest.fixture
def checks(main, tmp_path, monkeypatch):
    """A workflow whose check functions are stubbed; `state` controls them."""
    main, _ = main
    state = {'runs': [{'label': 'task-a', 'fieldmap': {}}, {'label': 'task-b', 'fieldmap': None}],
             'outputs': {'runs': {'task-a': {'n_frames': 5, 'outputs': {}}, 'task-b': {'n_frames': 5, 'outputs': {}}},
                         'errors': [], 'warnings': ['x: 2 vertices with low coverage']},
             'sdc': dict(SDC), 'calls': []}

    def result(key):
        if isinstance(state[key], Exception):
            raise state[key]
        return state[key]

    def find_runs(*args, **kwargs):
        state['calls'].append(('find_runs', args, kwargs))
        return result('runs')

    def check_outputs(**kwargs):
        state['calls'].append(('check_outputs', kwargs))
        return result('outputs')

    def check_sdc(run, bids_dir, n_frames):
        state['calls'].append(('check_sdc', run['label'], bids_dir, n_frames))
        return result('sdc')
    monkeypatch.setattr(main, 'find_runs', find_runs)
    monkeypatch.setattr(main, 'check_outputs', check_outputs)
    monkeypatch.setattr(main, 'check_sdc', check_sdc)
    wf = _workflow(main, tmp_path, '25.2.7', '25.2.7')

    def load():
        with open(tmp_path / 'nb-data' / 'checks' / 'sub-01.json') as f:
            return json.load(f, parse_constant=pytest.fail)
    return wf, state, load


def test_check(checks, tmp_path, capsys):
    wf, state, load = checks
    assert wf.check(filter_=len, no_fieldmap=('task-b',), fieldmap_prefix={'a': 'b'}, n_frames=7,
                    n_jobs=4, n_depths=5, low_coverage=0.4) is True
    out = tmp_path / 'fmriprep_out'
    args = dict(
        sid='01', bids_dir=str(tmp_path / 'bids'), fmriprep_dir=str(out),
        out_dir=str(tmp_path / 'nb-data' / 'resampled'), xform_dir=str(tmp_path / 'nb-data' / 'xforms'),
        coverage_dir=str(tmp_path / 'nb-data' / 'confounds' / 'coverage'),
        combinations=[('onavg-ico32', '1step_pial_overlap')],
        fs_dir=str(out / 'sourcedata' / 'freesurfer' / 'sub-01'),
        low_coverage=0.4, runs=state['runs'])
    assert state['calls'] == [
        ('find_runs', (str(out), str(tmp_path / 'bids'), '01', len),
         {'no_fieldmap': ('task-b',), 'fieldmap_prefix': {'a': 'b'}}),
        ('check_outputs', args),
        ('check_sdc', 'task-a', str(tmp_path / 'bids'), 7)]
    report = load()
    assert report['sid'] == '01' and report['errors'] == []
    assert report['warnings'] == ['x: 2 vertices with low coverage', 'task-b: SDC not checked (no fieldmap)']
    assert report['runs']['task-a'] == {'n_frames': 5, 'outputs': {}, 'sdc': dict(SDC, status='ok')}
    assert report['runs']['task-b']['sdc'] == {'status': 'skipped'}
    assert 'Warning: task-b: SDC not checked' in capsys.readouterr().out
    # No log files: the step runs again
    assert not list((tmp_path / 'archive' / 'logs').iterdir())
    state['calls'].clear()
    assert wf.check() is True and len(state['calls']) == 3


def test_check_sdc_not_improved(checks):
    wf, state, load = checks
    state['sdc'] = dict(SDC, r_sdc=0.8)
    assert wf.check() is True
    report = load()
    assert report['runs']['task-a']['sdc']['status'] == 'not_improved'
    assert report['errors'] == [] and 'task-a: SDC does not improve' in report['warnings'][1]


@pytest.mark.parametrize('sdc, expected, message', [
    (dict(SDC, r_sdc=float('nan')), dict(SDC, r_sdc=None), 'task-a: SDC check has nonfinite values'),
    (dict(SDC, shift_p99=float('inf')), dict(SDC, shift_p99=None), 'task-a: SDC check has nonfinite values'),
    (FileNotFoundError('no reference'), {}, 'task-a: SDC check failed (FileNotFoundError: no reference)'),
])
def test_check_sdc_inconclusive(checks, capsys, sdc, expected, message):
    wf, state, load = checks
    state['sdc'] = sdc
    with pytest.raises(SystemExit) as e:
        wf.check()
    assert e.value.code == 1
    report = load()
    assert report['errors'] == [message]
    assert report['runs']['task-a']['sdc'] == dict(expected, status='inconclusive')
    assert report['runs']['task-b']['sdc'] == {'status': 'skipped'}
    assert f'Error: {message}' in capsys.readouterr().out


def test_check_errors(checks):
    wf, state, load = checks
    # Errors in the files fail the step after the report is written, and can be repaired
    state['outputs']['errors'] = ['x: missing']
    with pytest.raises(SystemExit):
        wf.check()
    assert load()['errors'] == ['x: missing'] and load()['runs']['task-a']['sdc']['status'] == 'ok'
    state['outputs']['errors'] = []
    assert wf.check() is True and load()['errors'] == []
    # Runs that cannot be found
    state['runs'] = ValueError('no transform')
    with pytest.raises(SystemExit):
        wf.check()
    assert load() == {'sid': '01', 'runs': {}, 'warnings': [],
                      'errors': ['cannot find the runs (ValueError: no transform)']}


def test_check_unsupported_arguments(checks, tmp_path):
    wf, state, load = checks
    for kwargs in [{'runs': []}, {'wf_root': 'x'}]:
        with pytest.raises(TypeError, match='Unsupported arguments'):
            wf.check(**kwargs)
    assert not state['calls'] and not (tmp_path / 'nb-data' / 'checks').exists()


def test_check_versions(main, tmp_path):
    main, _ = main
    with pytest.raises(NotImplementedError, match='25 or later'):
        _workflow(main, tmp_path / 'a', '24.1.1', '24.1.1').check()
    with pytest.raises(ValueError, match='configured'):
        _workflow(main, tmp_path / 'b', '25.2.7', '24.1.1').check()
    assert not (tmp_path / 'a' / 'nb-data' / 'checks').exists()

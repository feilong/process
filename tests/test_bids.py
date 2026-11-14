import json

import pytest

from process.bids import parse_bids_filename, get_bids_metadata


def test_parse_bids_filename():
    assert parse_bids_filename('/a/sub-01_ses-2_task-rest_run-1_bold.nii.gz') == {
        'sub': '01', 'ses': '2', 'task': 'rest', 'run': '1',
        'suffix': 'bold', 'ext': '.nii.gz'}
    assert parse_bids_filename('task-rest_bold.json') == {
        'task': 'rest', 'suffix': 'bold', 'ext': '.json'}


def _write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


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

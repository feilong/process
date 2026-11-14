import json

import nibabel as nib
import numpy as np
import pytest


def _write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


@pytest.fixture
def bids(tmp_path):
    """A small BIDS dataset with inherited sidecars; returns (root, raw BOLD path)."""
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
    nib.save(nib.Nifti1Image(np.zeros((10, 64, 3), dtype=np.float32), np.eye(4)), bold)
    return root, str(bold)

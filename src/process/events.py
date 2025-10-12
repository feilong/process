import os
import shutil
from glob import glob


def events_workflow(bids_dir, events_dir, filter_=None):
    in_fns = sorted(glob(os.path.join(bids_dir, 'sub-*', 'func', '*events.tsv'))) + \
        sorted(glob(os.path.join(bids_dir, 'sub-*', 'ses-*', 'func', '*events.tsv')))
    if filter_ is not None:
        in_fns = filter_(in_fns)
    for in_fn in in_fns:
        out_fn = os.path.join(events_dir, os.path.basename(in_fn))
        shutil.copy2(in_fn, out_fn)

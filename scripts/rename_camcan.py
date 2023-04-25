import os
import shutil
from glob import glob
import filecmp
import json


if __name__ == '__main__':
    """
    `dataset_description.json` and `README` are identical across sub-folders.
    `participants.tsv` of other sub-folders are a subset of the one in `anat`.
    - 653 with anatomical data
    - 652 with rest data
    - 649 with smt data
    - 649 with movie data
    """
    shared_files = [
        # 'CHANGES',
        'dataset_description.json',
        # 'participants.tsv',
        'README',
    ]
    in_root = os.path.expanduser(
        '~/lab/camcan_new/cc700/mri/pipeline/release004/BIDS_20190411')
    out_root = os.path.expanduser('~/lab/BIDS/camcan')
    os.makedirs(out_root, exist_ok=True)

    sections = ['anat', 'epi_movie', 'epi_rest', 'epi_smt', 'fmap_movie', 'fmap_rest', 'fmap_smt']
    for section in sections[1:]:
        for basename in shared_files:
            fn1 = os.path.join(in_root, 'anat', basename)
            fn2 = os.path.join(in_root, section, basename)
            res = filecmp.cmp(fn1, fn2)
            if not res:
                print(res, fn1, fn2)

    basename = 'participants.tsv'
    fn1 = os.path.join(in_root, 'anat', basename)
    with open(fn1, 'r') as f:
        lines1 = f.read().splitlines()
    for section in sections[1:]:
        fn2 = os.path.join(in_root, section, basename)
        with open(fn2, 'r') as f:
            lines2 = f.read().splitlines()
            for line in lines2:
                assert line in lines1

    for basename in shared_files + ['participants.tsv']:
        fn = os.path.join(in_root, 'anat', basename)
        out_fn = os.path.join(out_root, basename)
        if not os.path.exists(out_fn):
            shutil.copy2(fn, out_fn)

    sids = [os.path.basename(_) for _ in sorted(glob(f'{in_root}/anat/sub-*'))]
    print(len(sids))

    ## anat
    for sid in sids:
        in_dir = f'{in_root}/anat/{sid}/anat'
        out_dir = f'{out_root}/{sid}/anat'
        os.makedirs(out_dir, exist_ok=True)
        for fn in sorted(glob(f'{in_dir}/*.*')):
            out_fn = os.path.join(out_dir, os.path.basename(fn))
            if not os.path.exists(out_fn):
                shutil.copy2(fn, out_fn)

    ## movie
    for sid in sids:
        in_dir = f'{in_root}/epi_movie/{sid}/epi_movie'
        if not os.path.exists(in_dir):
            if sid[4:] not in ['CC110062', 'CC410129', 'CC610050', 'CC710214']:
                print(in_dir)
            continue
        out_dir = f'{out_root}/{sid}/func'
        os.makedirs(out_dir, exist_ok=True)
        fns = sorted(glob(f'{in_dir}/*.*'))

        # Assert we have exactly 5 echoes of movie data
        assert len(fns) == 10
        for echo in ['echo1', 'echo2', 'echo3', 'echo4', 'echo5']:
            for ext in ['json', 'nii.gz']:
                fn = os.path.join(in_dir, f'{sid}_epi_movie_{echo}.{ext}')
                assert fn in fns
                out_fn = os.path.join(out_dir, f'{sid}_task-bang_run-01_{echo}.{ext}')
                if not os.path.exists(out_fn):
                    shutil.copy2(fn, out_fn)

    ## movie fmap
    for sid in sids:
        in_dir = f'{in_root}/fmap_movie/{sid}/fmap'
        if not os.path.exists(in_dir):
            if sid[4:] not in ['CC110062', 'CC410129', 'CC610050', 'CC710214']:
                print(in_dir)
            continue
        out_dir = f'{out_root}/{sid}/fmap'
        os.makedirs(out_dir, exist_ok=True)
        fns = sorted(glob(f'{in_dir}/*.*'))

        # one phasediff, two magnitude
        assert len(fns) == 6
        for ext in ['json', 'nii.gz']:
            for tag, tag2 in [
                    ('fmap', 'phasediff'),
                    ('run-01_fmap', 'magnitude1'),  # shorter echo
                    ('run-02_fmap', 'magnitude2'),  # longer echo
                ]:
                fn = os.path.join(in_dir, f'{sid}_{tag}.{ext}')
                assert fn in fns
                out_fn = os.path.join(out_dir, f'{sid}_acq-bang_{tag2}.{ext}')
                if os.path.exists(out_fn):
                    continue

                if ext == 'nii.gz':
                    shutil.copy2(fn, out_fn)
                elif ext == 'json':
                    with open(fn, 'r') as f:
                        d = json.load(f)
                    if tag2 == 'phasediff':
                        d['EchoTime1'] = 0.00519
                        d['EchoTime2'] = 0.00765
                        d['IntendedFor'] = [f'func/{sid}_task-bang_run-01_echo{e}.nii.gz'
                            for e in [1, 2, 3, 4, 5]]
                        with open(out_fn, 'w') as f:
                            json.dump(d, f)
                    else:
                        assert d['EchoTime'] == {
                            'magnitude1': 0.00519,
                            'magnitude2': 0.00765,
                            }[tag2]
                        shutil.copy2(fn, out_fn)

    ## rest and smt
    for sid in sids:
        out_dir = f'{out_root}/{sid}/func'
        os.makedirs(out_dir, exist_ok=True)
        for task in ['rest', 'smt']:
            in_dir = f'{in_root}/epi_{task}/{sid}/epi_{task}'
            if not os.path.exists(in_dir):
                if (sid[4:] not in ['CC110062'] and
                        (task == 'smt' and sid[4:] not in ['CC220519', 'CC610462', 'CC710518'])):
                    print(in_dir)
                continue
            fns = sorted(glob(f'{in_dir}/*.*'))

            # Assert we have exactly one run each
            if task == 'rest':
                assert len(fns) == 2
            else:
                if len(fns) != 3 and sid[4:] not in ['CC120065', 'CC221755', 'CC410222', 'CC610146']:
                    print(in_dir)
                    print(fns)
                    assert len(fns) == 3

            for ext in ['json', 'nii.gz']:
                fn = os.path.join(in_dir, f'{sid}_epi_{task}.{ext}')
                assert fn in fns
                out_fn = os.path.join(out_dir, f'{sid}_task-{task}_run-01_bold.{ext}')
                if not os.path.exists(out_fn):
                    shutil.copy2(fn, out_fn)

    ## rest and smt fmap
    for sid in sids:
        out_dir = f'{out_root}/{sid}/fmap'
        os.makedirs(out_dir, exist_ok=True)
        for task in ['rest', 'smt']:
            in_dir = f'{in_root}/fmap_{task}/{sid}/fmap'
            if not os.path.exists(in_dir):
                if (sid[4:] not in ['CC110062'] and
                        (task == 'smt' and sid[4:] not in ['CC710518'])):
                    print(in_dir)
                continue
            fns = sorted(glob(f'{in_dir}/*.*'))

            # one phasediff, two magnitude
            assert len(fns) == 6
            for ext in ['json', 'nii.gz']:
                for tag, tag2 in [
                        ('fmap', 'phasediff'),
                        ('run-01_fmap', 'magnitude1'),  # shorter echo
                        ('run-02_fmap', 'magnitude2'),  # longer echo
                    ]:
                    fn = os.path.join(in_dir, f'{sid}_{tag}.{ext}')
                    assert fn in fns
                    out_fn = os.path.join(out_dir, f'{sid}_acq-{task}_{tag2}.{ext}')
                    if os.path.exists(out_fn):
                        continue

                    if ext == 'nii.gz':
                        shutil.copy2(fn, out_fn)
                    elif ext == 'json':
                        with open(fn, 'r') as f:
                            d = json.load(f)
                        if tag2 == 'phasediff':
                            d['EchoTime1'] = 0.00519
                            d['EchoTime2'] = 0.00765
                            d['IntendedFor'] = [f'func/{sid}_task-{task}_run-01_bold.nii.gz'
                                for e in [1, 2, 3, 4, 5]]
                            with open(out_fn, 'w') as f:
                                json.dump(d, f)
                        else:
                            assert d['EchoTime'] == {
                                'magnitude1': 0.00519,
                                'magnitude2': 0.00765,
                                }[tag2]
                            shutil.copy2(fn, out_fn)

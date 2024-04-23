import os


def t2smap_cmd(config, wf_dir='cd /dartfs/rc/lab/H/HaxbyLab/feilong/fmriprep_work_root/camcan_20.2.7/work_CC110033/fmriprep_wf/single_subject_CC110033_wf/func_preproc_task_bang_run_01_echo_1_wf/bold_t2smap_wf/t2smap_node/'):
    if not os.path.exists(wf_dir):
        for e in range(1, 10):
            wf_dir_ = wf_dir.replace('_echo_1_', f'_echo_{e}_')
            if os.path.exists(wf_dir_):
                wf_dir = wf_dir_
                break

    cmd = [
        'singularity', 'exec', '-e',
        *config['singularity_options'], '-H', config['singularity_home'],
        config['singularity_image'], '/bin/bash', '-c', 
        f"cd {wf_dir}/bold_t2smap_wf/t2smap_node/; . command.txt;",
    ]
    return cmd

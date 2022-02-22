import os
from itertools import product
from care_batch.care_prep import care_prep
from care_batch.datagen import datagen
from care_batch.evaluate import evaluate, summarize_stats
from care_batch.restore import restore
from care_batch.train import train

from cosmos.api import (
    Cosmos,
    Dependency,
    draw_stage_graph,
    draw_task_graph,
    pygraphviz_available,
    py_call,
)
from cosmos.util.helpers import environment_variables



def main():
    
    base_dir = "/research/sharedresources/cbi/data_exchange/hangrp/Development/Restoration/Measured/Analysis_test"

    parameter_fn = 'parameters.json'

    db_filename = "/research/sharedresources/cbi/data_exchange/hangrp/"\
        "Development/Restoration/Measured/Analysis_test/cosmos.sqlite"
    workflow_name = 'CARE_test'

    base_dir = "/research/sharedresources/cbi/data_exchange/hangrp/"\
        "Development/Restoration/Measured/Analysis_test"
    input_dir = "ROI"
    data_dir = "CARE_data"
    model_dir = 'CARE_models'
    accuracy_dir = 'accuracy'
    accuracy_fn = 'accuracy.csv'
    n_jobs = 30
    name_high='high'
    name_low='low'
    axes = 'ZYX'

    input_pairs = [('high/deconvolved', 'low/raw'),
                   ('high/deconvolved', 'low/deconvolved')]
    patch_size = [(8, 8, 8),
                  (16, 16, 16)]
    n_patches_per_image = [50]

    train_epochs = 10
    train_steps_per_epoch = [10]
    train_batch_size = [8]

    cosmos = Cosmos("cosmos.sqlite", default_drm='local', default_max_attempts=2)
    cosmos.initdb()

    workflow = cosmos.start("CARE_test", skip_confirm=True)

    care_prep_tasks = []
    for input_pair in input_pairs:
        uid = rf"{input_pair[0].replace('/', '_')}_vs_{input_pair[1].replace('/', '_')}"
        task = workflow.add_task(
            func=care_prep, 
            params=dict(input_pair=[os.path.join(base_dir, input_dir, fn) for fn in input_pair], 
                        output_dir=os.path.join(base_dir, data_dir, uid),
                        name_high=name_high, 
                        name_low=name_low), 
            uid=uid)
        care_prep_tasks.append(task)

    datagen_tasks = []
    for care_prep_task in care_prep_tasks:
        pair_dir = care_prep_task.params['output_dir']
        for ps, n_patches in product(patch_size, n_patches_per_image):
            uid = rf"{pair_dir.split('/')[-1]}_patch_size={ps}_npatches={n_patches}"
            task = workflow.add_task(
                func=datagen,
                params=dict(basepath=pair_dir, save_file=os.path.join(base_dir, data_dir, rf"{uid}.npz"), 
                            target_dir=name_high, source_dir=name_low, axes=axes,
                            patch_size=ps,
                            n_patches_per_image=n_patches),
                parents=[care_prep_task],
                uid=uid
            )
            datagen_tasks.append(task)
            
    train_tasks = []
    for datagen_task in datagen_tasks:
        data_file = datagen_task.params['save_file']
        for st, bs in product(train_steps_per_epoch, train_batch_size):
            uid = rf"{data_file[:-4].split('/')[-1]}_steps_per_epoch={st}_batchsize={bs}"
            task = workflow.add_task(
                func=train,
                params=dict(data_file=data_file, model_basedir=os.path.join(base_dir, model_dir),
                            train_steps_per_epoch=st, train_batch_size=bs,
                            model_name=uid),
                parents=[datagen_task],
                uid=uid,
                gpu_req=1
            )
            train_tasks.append(task)
            
    restore_tasks = []
    for care_prep_task in care_prep_tasks:
        for train_task in train_tasks:
            input_dir = care_prep_task.params['output_dir']
            model_name = train_task.params['model_name']
            uid = rf"{input_dir.split('/')[-1]}_{model_name}"
            output_dir = os.path.join(input_dir, rf"{name_low}_restored_{model_name}")
            task = workflow.add_task(
                func=restore,
                params=dict(input_dir=os.path.join(input_dir, name_low),
                            output_dir=output_dir, model_basedir=train_task.params['model_basedir'],
                            model_name=model_name, axes=axes),
                parents=[care_prep_task, train_task],
                uid=uid,
                gpu_req=1
            )
            restore_tasks.append(task)
            
    evaluation_tasks = []
    for restore_task in restore_tasks:
        input_dir = restore_task.params['output_dir']
        uid = input_dir.split('/')[-2] + '_' + input_dir.split('/')[-1]
        task = workflow.add_task(
            func=evaluate,
            params=dict(input_dir=input_dir, gt_dir=os.path.join(input_dir, rf'../{name_high}'),
                        output_fn=os.path.join(base_dir, accuracy_dir, uid + '.csv'),
                        model_name=restore_task.params['model_name'], pair_name=input_dir.split('/')[-2]),
            uid=uid,
            parents=[restore_task]
        )
        evaluation_tasks.append(task)

    summarize_task = workflow.add_task(
        func=summarize_stats,
        params=dict(input_fns=[t.params['output_fn'] for t in evaluation_tasks], 
                    output_fn=os.path.join(base_dir, accuracy_fn)),
        parents=evaluation_tasks,
        uid="",
    ) 
            

    workflow.make_output_dirs()
    workflow.run(max_cores=20, cmd_wrapper=py_call, max_gpus=2)

    
if __name__ == "__main__":
    with environment_variables(COSMOS_LOCAL_GPU_DEVICES="0,1"):
        main()
import os
import json
import argparse
from itertools import product
from care_batch.care_prep import care_prep
from care_batch.datagen import datagen
from care_batch.evaluate import evaluate, summarize_stats
from care_batch.restore import restore
from care_batch.train import train

from cosmos.api import (
    Cosmos,
    py_call,
)
from cosmos.util.helpers import environment_variables


def set_care_prep_tasks(workflow, params):
    care_prep_tasks = []
    for input_pair in params.input_pairs:
        uid = rf"{input_pair[0].replace('/', '_')}_vs_{input_pair[1].replace('/', '_')}"
        task = workflow.add_task(
            func=care_prep,
            params=dict(input_pair=[os.path.join(params.base_dir, params.input_dir, fn)
                                    for fn in input_pair],
                        output_dir=os.path.join(params.base_dir, params.data_dir, uid),
                        name_high=params.name_high,
                        name_low=params.name_low),
            uid=uid)
        care_prep_tasks.append(task)
    return care_prep_tasks


def set_datagen_tasks(workflow, care_prep_tasks, params):
    datagen_tasks = []
    for care_prep_task in care_prep_tasks:
        pair_dir = care_prep_task.params['output_dir']
        for ps, n_patches in product(params.patch_size, params.n_patches_per_image):
            uid = rf"{pair_dir.split('/')[-1]}_patch_size={ps}_npatches={n_patches}"
            task = workflow.add_task(
                func=datagen,
                params=dict(basepath=pair_dir,
                            save_file=os.path.join(params.base_dir, params.data_dir, rf"{uid}.npz"),
                            target_dir=params.name_high,
                            source_dir=params.name_low,
                            axes=params.axes,
                            patch_size=ps,
                            n_patches_per_image=n_patches),
                parents=[care_prep_task],
                uid=uid
            )
            datagen_tasks.append(task)
    return datagen_tasks


def get_train_tasks(workflow, datagen_tasks, params):
    train_tasks = []
    for datagen_task in datagen_tasks:
        data_file = datagen_task.params['save_file']
        for st, bs in product(params.train_steps_per_epoch, params.train_batch_size):
            uid = rf"{data_file[:-4].split('/')[-1]}_steps_per_epoch={st}_batchsize={bs}"
            task = workflow.add_task(
                func=train,
                params=dict(data_file=data_file,
                            model_basedir=os.path.join(params.base_dir, params.model_dir),
                            train_steps_per_epoch=st,
                            train_batch_size=bs,
                            model_name=uid),
                parents=[datagen_task],
                uid=uid,
                gpu_req=1
            )
            train_tasks.append(task)
    return train_tasks


def get_restore_tasks(workflow, care_prep_tasks, train_tasks, params):
    restore_tasks = []
    for care_prep_task in care_prep_tasks:
        for train_task in train_tasks:
            input_dir = care_prep_task.params['output_dir']
            model_name = train_task.params['model_name']
            uid = rf"{input_dir.split('/')[-1]}_{model_name}"
            output_dir = os.path.join(input_dir, rf"{params.name_low}_restored_{model_name}")
            task = workflow.add_task(
                func=restore,
                params=dict(input_dir=os.path.join(input_dir, params.name_low),
                            output_dir=output_dir,
                            model_basedir=train_task.params['model_basedir'],
                            model_name=model_name,
                            axes=params.axes),
                parents=[care_prep_task, train_task],
                uid=uid,
                gpu_req=1
            )
            restore_tasks.append(task)
    return restore_tasks


def get_evaluation_tasks(workflow, restore_tasks, params):
    evaluation_tasks = []
    for restore_task in restore_tasks:
        input_dir = restore_task.params['output_dir']
        uid = input_dir.split('/')[-2] + '_' + input_dir.split('/')[-1]
        task = workflow.add_task(
            func=evaluate,
            params=dict(input_dir=input_dir,
                        gt_dir=os.path.join(input_dir, rf'../{params.name_high}'),
                        output_fn=os.path.join(params.base_dir, params.accuracy_dir, uid + '.csv'),
                        model_name=restore_task.params['model_name'],
                        pair_name=input_dir.split('/')[-2]),
            uid=uid,
            parents=[restore_task]
        )
        evaluation_tasks.append(task)
    return evaluation_tasks


def get_summarize_task(workflow, evaluation_tasks, params):
    summarize_task = workflow.add_task(
        func=summarize_stats,
        params=dict(input_fns=[t.params['output_fn'] for t in evaluation_tasks],
                    output_fn=os.path.join(params.base_dir, params.accuracy_fn)),
        parents=evaluation_tasks,
        uid="",
    )
    return summarize_task


def recipe(workflow, params):
    care_prep_tasks = set_care_prep_tasks(workflow, params)
    datagen_tasks = set_datagen_tasks(workflow, care_prep_tasks, params)
    train_tasks = get_train_tasks(workflow, datagen_tasks, params)
    restore_tasks = get_restore_tasks(workflow, care_prep_tasks, train_tasks, params)
    evaluation_tasks = get_evaluation_tasks(workflow, restore_tasks, params)
    get_summarize_task(workflow, evaluation_tasks, params)


def main():

    p = argparse.ArgumentParser()
    p.add_argument("-drm", default="local", help="", choices=("local", "awsbatch", "slurm", "drmaa:ge", "ge"))
    p.add_argument("-q", "--queue", help="Submit to this queue if the DRM supports it")
    p.add_argument("-p", "--parameter-file", help="Parameter file name")

    args = p.parse_args()
    with open(args.parameter_file) as f:
        params = json.load(f)
    params = argparse.Namespace(**params)

    cosmos = Cosmos(params.db_filename, default_drm=args.drm, default_max_attempts=2, default_queue=args.queue)
    cosmos.initdb()

    workflow = cosmos.start(params.workflow_name, skip_confirm=True)
    recipe(workflow, params)

    workflow.make_output_dirs()
    workflow.run(max_cores=params.n_jobs, cmd_wrapper=py_call, max_gpus=2)

    
if __name__ == "__main__":
    with environment_variables(COSMOS_LOCAL_GPU_DEVICES="0,1"):
        main()

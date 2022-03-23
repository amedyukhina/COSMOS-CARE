import os
import json
import argparse
from itertools import product
from am_utils.utils import walk_dir
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


def __copy_data(input_dir, output_dir):
    for fn in walk_dir(input_dir):
        fn_out = fn.replace(input_dir, output_dir)
        if not os.path.exists(fn_out):
            os.makedirs(os.path.dirname(fn_out), exist_ok=True)
            os.symlink(fn, fn_out)

            
def __get_subfolder(path):
    return path.rstrip('/').split('/')[-1]


def set_datagen_tasks(workflow, params):
    datagen_tasks = []
    for ps, n_patches in product(params.patch_size, params.n_patches_per_image):
        uid = rf"patch_size={ps}_npatches={n_patches}"
        basepath = os.path.abspath(os.path.join(params.output_dir,
                                                __get_subfolder(params.input_dir), params.name_train))
        task = workflow.add_task(
            func=datagen,
            params=dict(basepath=basepath,
                        save_file=os.path.join(params.output_dir, params.npz_dir, rf"{uid}.npz"),
                        target_dir=params.name_high,
                        source_dir=params.name_low,
                        axes=params.axes,
                        patch_size=ps,
                        n_patches_per_image=n_patches),
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
                            model_basedir=os.path.join(params.output_dir, params.model_dir),
                            train_steps_per_epoch=st,
                            train_batch_size=bs,
                            save_history=params.save_training_history,
                            model_name=uid),
                parents=[datagen_task],
                uid=uid,
                gpu_req=1
            )
            train_tasks.append(task)
    return train_tasks


def get_restore_tasks(workflow, train_tasks, params):
    restore_tasks = []
    input_dir = os.path.abspath(os.path.join(params.output_dir, __get_subfolder(params.input_dir), 
                                             params.name_validation, params.name_low))
    for train_task in train_tasks:
        model_name = train_task.params['model_name']
        uid = rf"restored_{model_name}"
        output_dir = input_dir + rf"_{uid}"
        task = workflow.add_task(
            func=restore,
            params=dict(input_dir=input_dir,
                        output_dir=output_dir,
                        model_basedir=train_task.params['model_basedir'],
                        model_name=model_name,
                        axes=params.axes),
            parents=[train_task],
            uid=uid,
            gpu_req=1
        )
        restore_tasks.append(task)
    return restore_tasks


def get_evaluation_tasks(workflow, restore_tasks, params):

    evaluation_tasks = []
    ids = []
    
    for restore_task in restore_tasks:
        input_dir = restore_task.params['output_dir']
        uid = input_dir.split('/')[-1]
        ids.append(uid)
        task = workflow.add_task(
            func=evaluate,
            params=dict(input_dir=input_dir,
                        gt_dir=os.path.join(input_dir, rf'../{params.name_high}'),
                        output_fn=os.path.join(params.output_dir, params.accuracy_dir, uid + '.csv'),
                        model_name=restore_task.params['model_name']),
            uid=uid,
            parents=[restore_task]
        )
        evaluation_tasks.append(task)
        
    
    base_dir = os.path.abspath(os.path.join(params.output_dir, __get_subfolder(params.input_dir), 
                                            params.name_validation))
    for inpdir in os.listdir(base_dir):
        if not inpdir in ids:
            input_dir = os.path.join(base_dir, inpdir)

            task = workflow.add_task(
                func=evaluate,
                params=dict(input_dir=input_dir,
                            gt_dir=os.path.join(input_dir, rf'../{params.name_high}'),
                            output_fn=os.path.join(params.output_dir, params.accuracy_dir, inpdir + '.csv'),
                            model_name=inpdir),
                uid=inpdir
            )
            evaluation_tasks.append(task)


    return evaluation_tasks


def get_summarize_task(workflow, evaluation_tasks, params):
    summarize_task = workflow.add_task(
        func=summarize_stats,
        params=dict(input_fns=[t.params['output_fn'] for t in evaluation_tasks],
                    output_fn=os.path.join(params.output_dir, params.accuracy_fn)),
        parents=evaluation_tasks,
        uid="",
    )
    return summarize_task


def recipe(workflow, params):
    datagen_tasks = set_datagen_tasks(workflow, params)
    train_tasks = get_train_tasks(workflow, datagen_tasks, params)
    restore_tasks = get_restore_tasks(workflow, train_tasks, params)
    evaluation_tasks = get_evaluation_tasks(workflow, restore_tasks, params)
    get_summarize_task(workflow, evaluation_tasks, params)


def main():

    p = argparse.ArgumentParser()
    p.add_argument("-drm", default="local", help="", choices=("local", "awsbatch", "slurm", "drmaa:ge", "ge"))
    p.add_argument("-q", "--queue", help="Submit to this queue if the DRM supports it")
    p.add_argument("-p", "--parameter-file", help="Parameter file name")
    p.add_argument("-g", "--n-gpus", type=int, default=2, help="Number of GPUs to use")
    p.add_argument("-c", "--n-cores", type=int, default=30, help="Number of CPU cores to use")

    args = p.parse_args()
    with open(args.parameter_file) as f:
        params = json.load(f)
    params = argparse.Namespace(**params)
    
    __copy_data(params.input_dir,
                os.path.join(params.output_dir, __get_subfolder(params.input_dir)))

    cosmos = Cosmos(params.db_filename, default_drm=args.drm, default_max_attempts=2, default_queue=args.queue)
    cosmos.initdb()

    workflow = cosmos.start(params.workflow_name, skip_confirm=True)
    recipe(workflow, params)

    workflow.make_output_dirs()
    os.makedirs(params.base_dir, exist_ok=True)
    workflow.run(max_cores=args.n_cores, cmd_wrapper=py_call, max_gpus=args.n_gpus)
    
if __name__ == "__main__":
    with environment_variables(COSMOS_LOCAL_GPU_DEVICES="0,1"):
        main()

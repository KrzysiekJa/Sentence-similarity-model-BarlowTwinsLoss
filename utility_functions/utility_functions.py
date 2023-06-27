import os
import shutil
import random
import numpy as np
import torch
import neptune


def set_seeds(seed: int):
    # Setting all seeds to make results reproducible
    torch.manual_seed( seed )
    torch.cuda.manual_seed_all( seed )
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed( seed )
    random.seed( seed )
    os.environ['PYTHONHASHSEED'] = str( seed )


def init_learning_env(name: str, tags_list: list):
    torch.cuda.empty_cache()
    run = neptune.init_run(
        capture_hardware_metrics=True,
        capture_stderr=True,
        capture_stdout=True
    )
    run["sys/name"] = name
    run["sys/tags"].add(tags_list)
    print( os.system('nvidia-smi') )
    return run


def set_neptun_params(run, params_dict: dict):
    run["model/name"] = params_dict['model_name']
    run["parameters"] = params_dict['params']
    run["parameters/barlow_twins_lambda"] = params_dict['lambda']
    run["parameters/batch_size"] = params_dict['batch_size']
    run["parameters/number_of_epochs"] = params_dict['num_epochs']
    run["dataset/name"] = params_dict['dataset_name']
    run["dataset/language"] = params_dict['language']
    return run


def set_neptun_train_params(run, params_dict: dict):
    run["parameters/train_steps"] = params_dict['train_steps']
    run["parameters/evaluation_steps"] = params_dict['evaluation_steps']
    run["parameters/warmup_steps"] = params_dict['warmup_steps']
    return run


def set_neptun_time_perf(run, end, start):
    run["train/time_perf_seconds"].append( round(end - start, 2) )
    run["train/time_perf_minutes"].append( round((end - start)/60, 3) )
    return run


def neptun_final_steps(run, test_evaluation, language: str, model_save_path: str):    
    project_read_only = neptune.init_project(
        mode="read-only"
    )
    
    run_pandas_df = project_read_only.fetch_runs_table(tag=["similarity", language]).to_pandas()
    best_testing_result = run_pandas_df["test/test_accuracy"].iloc[0] if "test/test_accuracy" in run_pandas_df else -1
    
    if test_evaluation > best_testing_result:
        run["model/model"].upload(model_save_path)
    shutil.rmtree(model_save_path)
    
    run.stop()


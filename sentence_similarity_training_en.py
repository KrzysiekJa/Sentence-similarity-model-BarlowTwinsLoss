import os
import shutil
import gzip
import csv
import math
import time
import random
import numpy as np
from datetime import datetime

import neptune
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from sentence_transformers import util, InputExample
from sentence_transformers.evaluation import SimilarityFunction, EmbeddingSimilarityEvaluator

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    os.system('pip install tensorboardX')
    from tensorboardX import SummaryWriter

from sentence_transformer import SentenceTransformer
from evaluators import LossEvaluator
from losses import BarlowTwinsLoss


os.environ["NEPTUNE_PROJECT"] = "kjarek/tests"
run = neptune.init_run(
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3ZDBhYTUwZS0yYmI5LTQyMmEtYmEwYi1iNjFlMzUyYjY1NGMifQ==",
    capture_hardware_metrics=True,
    capture_stderr=True,
    capture_stdout=True
)



def init_learning_env( run ):
    torch.cuda.empty_cache()
    os.system('nvidia-smi')
    run["sys/name"] = "basic-colab-example"
    run["sys/tags"].add(["colab", "tests", "similarity", "en"])


def set_seeds(seed: int):
    # Setting all seeds to make results reproducible
    torch.manual_seed( seed )
    torch.cuda.manual_seed_all( seed )
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed( seed )
    random.seed( seed )
    os.environ['PYTHONHASHSEED'] = str( seed )


def main( run ):
    ########################################################################
    # Checking if dataset exsist. If not, needed to download and extract
    ########################################################################
    sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

    if not os.path.exists(sts_dataset_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)
    
    ########################################################################
    # Training parameters
    ########################################################################
    model_name = 'nli-distilroberta-base-v2' # 'nli-distilroberta-base-v2' 'microsoft/deberta-base' 'allegro/herbert-base-cased'
    lambda_    = 5e-1
    batch_size = 32
    num_epochs = 12
    model_save_path = 'output/fine_tuning_benchmark-'+model_name.replace('/', '_')+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    run["model/name"] = model_name
    params = {
        "optimizer": {
            "type": "AdamW",
            "lr": 2e-5,
            "eps": 1e-12,
        }, 
    }
    run["parameters"] = params
    run["parameters/barlow_twins_lambda"] = lambda_
    run["parameters/batch_size"] = batch_size
    run["parameters/number_of_epochs"] = num_epochs
    run["dataset/name"] = sts_dataset_path[:-7]
    run["dataset/language"] = 'en'
    
    ########################################################################
    # Loading a pre-trained sentence transformer model
    ########################################################################
    model = SentenceTransformer(model_name)
    
    ########################################################################
    # Loading and preparing data
    ########################################################################
    train_samples = []
    dev_samples = []
    test_samples = []
    
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

            if row['split'] == 'dev':
                dev_samples.append(inp_example)
            elif row['split'] == 'test':
                test_samples.append(inp_example)
            else:
                train_samples.append(inp_example)
    
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
    log_dir = 'output/logs'
    dev_evaluator = LossEvaluator(dev_samples, loss_model=train_loss, log_dir=log_dir, show_progress_bar=True, batch_size=batch_size)
    
    ########################################################################
    # Configuring the training parameters
    ########################################################################
    train_loss = BarlowTwinsLoss(model=model, lambda_=lambda_)
    
    def neptune_callback(score, epoch, steps):
        global run
        run[f"epochs_val/val_loss"].append(score)
    
    evaluation_steps = len(train_dataloader) // 10
    warmup_steps = math.ceil( len(train_dataloader) * num_epochs * 0.1 )
    
    run["parameters/train_steps"] = len(train_dataloader)
    run["parameters/evaluation_steps"] = evaluation_steps
    run["parameters/warmup_steps"] = warmup_steps
    
    ########################################################################
    # Model training
    ########################################################################
    start = time.perf_counter()

    model.fit(
              train_objectives=[(train_dataloader, train_loss)],
              evaluator=dev_evaluator,
              epochs=num_epochs,
              evaluation_steps=evaluation_steps,
              show_progress_bar=True,
              warmup_steps=warmup_steps,
              optimizer_params={'lr': params['optimizer']['lr'], 'eps': params['optimizer']['eps']},
              callback=neptune_callback,
              output_path=model_save_path,
              training_samples=train_samples
    )

    end = time.perf_counter()
    
    run["train/time_perf_seconds"].append( round(end - start, 2) )
    run["train/time_perf_minutes"].append( round((end - start)/60, 3) )
    
    ########################################################################
    # Testing process
    ########################################################################
    model = SentenceTransformer(model_save_path)
    
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, main_similarity=SimilarityFunction.COSINE)
    test_evaluation = test_evaluator(model, output_path=model_save_path)
    run["test/test_accuracy"].append(test_evaluation)
    
    project_read_only = neptune.init_project(
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3ZDBhYTUwZS0yYmI5LTQyMmEtYmEwYi1iNjFlMzUyYjY1NGMifQ==",
        mode="read-only"
    )
    
    run_pandas_df = project_read_only.fetch_runs_table(tag=["similarity", "en"]).to_pandas()
    best_testing_result = run_pandas_df["test/test_accuracy"].iloc[0]
    
    if test_evaluation > best_testing_result:
        run["model/model"].upload(model_save_path)
    shutil.rmtree(model_save_path)
    
    run.stop()



if __name__ =='__main__':
    seed = 12 # on basis of: https://arxiv.org/pdf/2002.06305.pdf
    init_env( run )
    set_seeds( seed )
    main( run )




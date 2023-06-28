import os
import gzip
import csv
import math
import time
import numpy as np
from datetime import datetime

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
from utility_functions import *



def main( run, language: str ):
    ########################################################################
    # Checking if dataset exsist. If not, needed to download and extract
    ########################################################################
    sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

    if not os.path.exists(sts_dataset_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)
    ########################################################################
    # Training parameters
    ########################################################################
    model_name = os.environ.get("MODEL_NAME") # 'nli-distilroberta-base-v2' 'microsoft/deberta-base' 'allegro/herbert-base-cased'
    lambda_    = float( os.environ.get("LAMBDA_") )
    batch_size = 32
    num_epochs = 6
    model_save_path = 'output/fine_tuning_benchmark-'+model_name.replace('/', '_')+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    params = {
        "optimizer": {
            "type": "AdamW",
            "lr": 2e-5,
            "eps": 1e-12,
        }, 
    }
     
    run = set_neptun_params(run, 
        {
            "model_name": model_name,
            "params": params,
            "lambda": lambda_,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "dataset_name": sts_dataset_path[:-7],
            "language": language
        }
    )
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
    
    ########################################################################
    # Configuring training parameters and process objects
    ########################################################################
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
    log_dir = 'output/logs'
    train_loss = BarlowTwinsLoss(model=model, lambda_=lambda_)
    dev_evaluator = LossEvaluator(dev_samples, run, loss_model=train_loss, log_dir=log_dir, show_progress_bar=True, batch_size=batch_size)
    
    def neptune_callback(score, epoch, steps):
        global run
        run[f"epochs_val/val_loss"].append(score)
    
    evaluation_steps = len(train_dataloader) // 10
    warmup_steps = math.ceil( len(train_dataloader) * num_epochs * 0.1 )
    
    run = set_neptun_train_params(run,
        {
            "train_steps": len(train_dataloader),
            "evaluation_steps": evaluation_steps,
            "warmup_steps": warmup_steps
        }
    )
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
              training_samples=train_samples,
              run=run
    )
    end = time.perf_counter()
    
    run = set_neptun_time_perf(run, end, start)
    ########################################################################
    # Testing process
    ########################################################################
    model = SentenceTransformer(model_save_path)
    
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                test_samples, 
                main_similarity=SimilarityFunction.COSINE
    )
    test_evaluation = test_evaluator(model, output_path=model_save_path)
    
    run["test/test_accuracy"].append(test_evaluation)
    neptun_final_steps(run, test_evaluation, language, model_save_path)



if __name__ =='__main__':
    seed = 12 # on basis of: https://arxiv.org/pdf/2002.06305.pdf
    language = 'en'
    tags = ["athena", "slurm", "similarity", language]
    name = "slurm-execution-script-en"
    set_seeds( seed )
    run = init_learning_env( name, tags ) # returned: neptune.Run object
    main( run, language )



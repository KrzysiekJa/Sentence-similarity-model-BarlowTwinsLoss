import os
import gzip
import csv
import math
import torch
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers.integrations import NeptuneCallback # !!!
from datasets import load_dataset, load_metric

from utility_functions import set_seeds



seed = 12
set_seeds( seed )
torch.cuda.empty_cache()

########################################################################
# Training parameters
########################################################################

model_name = os.environ.get("MODEL_NAME")
batch_size = 32
num_epochs = 6
lr = 2e-5
w_decay = 0.01
eps = 1e-12

print('\nMODEL_NAME', model_name, '\n')
########################################################################
# Loading and preparing data
########################################################################

train_dataset = load_dataset('cdsc', 'cdsc-r', split='train[:-90%]') # last 90%
dev_dataset   = load_dataset('cdsc', 'cdsc-r', split='train[:10%]') # first 10%
test_dataset  = load_dataset('cdsc', 'cdsc-r', split='validation')

########################################################################
# Configuring training parameters and process objects
########################################################################

eval_steps = (len(train_dataset)/batch_size) // 10
warmup_steps = math.ceil( (len(train_dataset)/batch_size) * num_epochs * 0.1 )


model = AutoModelForSequenceClassification.from_pretrained( model_name, num_labels=1 )
# regression model, so num_labels=1


metric = load_metric('glue', 'stsb')

def compute_metrics( eval_predictions ):
    predictions, labels = eval_predictions
    predictions = predictions[:, 0]
    return metric.compute( predictions=predictions, references=labels )

neptune_callback = NeptuneCallback( log_parameters=False )
########################################################################
# Model training and testing
########################################################################

training_args = TrainingArguments(
    output_dir="output/test_trainer",
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    learning_rate=lr,
    weight_decay=w_decay,
    adam_epsilon=eps,
    eval_steps=eval_steps,
    warmup_steps=warmup_steps,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    save_total_limit=1,
    save_strategy='no',
    load_best_model_at_end=False,
    seed=seed,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset.shuffle(seed=seed),
    eval_dataset=dev_dataset.shuffle(seed=seed),
    compute_metrics=compute_metrics,
    callbacks=[ neptune_callback ]
)
trainer.train()

test_results = trainer.predict( test_dataset )

print('\nTest results:', test_results, '\n')

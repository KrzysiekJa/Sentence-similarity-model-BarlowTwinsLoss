import os
import gzip
import csv
import math
import torch
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers.integrations import NeptuneCallback # !!!
from datasets import Dataset, load_metric

from utility_functions import set_seeds



seed = 12
set_seeds( seed )
torch.cuda.empty_cache()

########################################################################
# Checking if dataset exsist. If not, needed to download and extract
########################################################################

sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

########################################################################
# Training parameters
########################################################################

model_name = os.environ.get("MODEL_NAME")
batch_size = 16
num_epochs = 6
lr = 2e-5
w_decay = 0.01
eps = 1e-12

print('\nMODEL_NAME', model_name, '\n')
########################################################################
# Loading and preparing data
########################################################################

train_samples = []
dev_samples = []
test_samples = []

with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score  = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        sample = dict(sentence1=row['sentence1'], sentence2=row['sentence2'], label=score)

        if row['split'] == 'dev':
            dev_samples.append( sample )
        elif row['split'] == 'test':
            test_samples.append( sample )
        else:
            train_samples.append( sample )

train_dataset = Dataset.from_list(train_samples)
dev_dataset  = Dataset.from_list(dev_samples)
test_dataset = Dataset.from_list(test_samples)
########################################################################
# Configuring training parameters and process objects
########################################################################

tokenizer = AutoTokenizer.from_pretrained( model_name )

def preprocess_function( examples ):
    return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

train_dataset = train_dataset.map( preprocess_function, batched=True )
dev_dataset  = dev_dataset.map( preprocess_function, batched=True )
test_dataset = test_dataset.map( preprocess_function, batched=True )


eval_steps = (len(train_dataset)/batch_size) // 10
warmup_steps = math.ceil( (len(train_dataset)/batch_size) * num_epochs * 0.1 )


model = AutoModelForSequenceClassification.from_pretrained( model_name, num_labels=1 )
# regression model, so num_labels=1


metric = load_metric('glue', 'stsb')

def compute_metrics( eval_predictions ):
    predictions, labels = eval_predictions
    if model_name.startswith("microsoft/deberta-"):
        predictions = predictions[:]
    else:
        predictions = predictions[:, 0]
    #cosine_scores = torch.cosine_similarity( predictions[0], predictions[1] )
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

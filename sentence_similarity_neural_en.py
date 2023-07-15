import os
import gzip
import csv
import math
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import DatasetDict, load_metric

from utility_functions import set_seeds



seed = 12
set_seeds( seed )

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
batch_size = 32
num_epochs = 6
lr = 2e-5
w_decay = 0.01
eps = 1e-12

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

datasets = DatasetDict({"train": train_samples, "dev": dev_samples, "test": test_samples})
########################################################################
# Configuring training parameters and process objects
########################################################################

tokenizer = AutoTokenizer.from_pretrained( model_name )

def preprocess_function( examples ):
    return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

encoded_datasets = datasets.map( preprocess_function, batched=True )


eval_steps = (len(train_dataset)/batch_size) // 10
warmup_steps = math.ceil( (len(train_dataset)/batch_size) * num_epochs * 0.1 )


model = AutoModelForSequenceClassification.from_pretrained( model_name, num_labels=1 )
# regression model, so num_labels=1


metric = load_metric('glue', 'stsb')

def compute_metrics( eval_predictions ):
    predictions, labels = eval_predictions
    predictions = predictions[:, 0]
    #cosine_scores = torch.cosine_similarity( predictions[0], predictions[1] )
    return metric.compute( predictions=cosine_scores, references=labels )

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
    report_to="neptune"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_datasets['train'].shuffle(seed=seed),
    eval_dataset=encoded_datasets['dev'].shuffle(seed=seed),
    compute_metrics=compute_metrics
)
trainer.train()

test_results = trainer.predict( encoded_datasets['test'] )


#!/bin/bash

#SBATCH --time=00:20:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

source python_venv/bin/activate
export NEPTUNE_API_TOKEN=""
export NEPTUNE_PROJECT="kjarek/sentence-similarity-research"
export LAMBDA_=$1
export MODEL_NAME="nli-distilroberta-base-v2"

python Sentence-similarity-model-Barlow-Twins/sentence_similarity_training_en.py

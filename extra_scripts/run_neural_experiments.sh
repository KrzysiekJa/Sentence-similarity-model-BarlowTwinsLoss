#!/bin/bash

rm slurm-*.out
rm -rf output/*
rm -rf .cache/
hpc-fs

for l in 0.01 0.005 0.001 0.0005 0.0001
do
    sbatch job.slurm "$l"
    sleep 5
done

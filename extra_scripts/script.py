import pandas as pd


df = pd.read_csv("train_val_accuracy.csv", header=None)

accuracy = df[2].tolist()
#warmup_steps, evaluation_steps = 108, 18 # <- eng
warmup_steps, evaluation_steps = 132, 22 # <- pl
warmup_evals = warmup_steps // evaluation_steps
accuracy = accuracy[warmup_evals:]
epochs = 6
epoch_len = int( len(accuracy)/epochs )


for i in range(epochs):
    print( str( accuracy[i * epoch_len + epoch_len - 1]) + ', ', end='' )

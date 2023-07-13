import time
import numpy as np
import torch
from torch import nn, Tensor
from typing import Iterable, Dict
import pickle


"""object model has been removed"""
# global vars
lambda_ = 5e-2
margin = 1e-12
loss_fct=nn.MSELoss()
score_transformation=nn.Identity()


# single v. ----------------------------------------------------------------------- #




# !not using at the moment!
def off_diagonal(x: Tensor) -> Tensor:
    n = x.size(0)
    flattened = x.flatten()[:-1]
    off_diagonals = flattened.reshape(n - 1, n + 1)[:, 1:]
    return off_diagonals.flatten()


def barlow_twins_loss_single(z_a: Tensor, z_b: Tensor) -> Tensor:
    """Compute the loss.
    Args:
        z_a: Embedding A
        z_b: Embedding B
    Returns:
        loss
    """
    if z_a.size() != z_b.size():
        raise ValueError("Embeddings error: different dimensions.")
    #z_a = (z_a - z_a.mean(0)) / z_a.std(0)
    #z_b = (z_b - z_b.mean(0)) / z_b.std(0)
    #z_a = nn.functional.normalize(z_a, p=1.0, dim=0)
    #z_b = nn.functional.normalize(z_b, p=1.0, dim=0)
    #z_a -= z_a.min()
    #z_a /= z_a.max()
    #z_b -= z_b.min()
    #z_b /= z_b.max()
    
    c = torch.matmul( z_a.reshape(-1,1), z_b.reshape(1,-1) )
    #c -= c.min()
    #c /= c.max()
    on_diag = torch.diagonal( (1.0 - c) ).pow(2)
    off_diag = c.fill_diagonal_(0).pow(2)
    
    on_diag  = (on_diag - on_diag.min()) / (on_diag.max() - on_diag.min())
    off_diag = (off_diag - off_diag.min()) / (off_diag.max() - off_diag.min())
    
    loss = on_diag.sum() + lambda_ * off_diag.sum() + margin
    
    return loss / z_a.size(-1)


def forward_single(sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
    embeddings = [sentence_feature['sentence_embedding'] for sentence_feature in sentence_features ]
    
    for i in range(len(embeddings[0])):
      out = barlow_twins_loss_single(embeddings[0][i], embeddings[1][i]).reshape(1)
      if i == 0:
          output = out
          continue
      output = torch.cat((output, out),0)
    #print(output, '\n', labels)
    return loss_fct( score_transformation(output), labels.view(-1) )

# batch v. ----------------------------------------------------------------------- #




def barlow_twins_loss(z_a: Tensor, z_b: Tensor) -> Tensor:
    """Compute the loss.
    Args:
        z_a: Embedding A
        z_b: Embedding B
    Returns:
        loss
    """
    if z_a.size() != z_b.size():
        raise ValueError("Embeddings error: different dimensions.")
    
    c = torch.bmm( z_a[:, :, None], z_b[:, None, :] ) # batch matrix-matrix product
    on_diag  = torch.diagonal( (1.0 - c), dim1=-2, dim2=-1 ).pow(2)
    off_diag = ( ( 1.0 - torch.eye(c.size(-1)) ) * c ).pow(2)   # device=c.device
    
    on_diag_min, on_diag_max  = on_diag.min(1).values[:, None] , on_diag.max(1).values[:, None]
    off_diag_min = (off_diag.min(2).values).min(1).values[:, None, None]
    off_diag_max = (off_diag.max(2).values).max(1).values[:, None, None]
    on_diag  = (on_diag  - on_diag_min)  / (on_diag_max  - on_diag_min)
    off_diag = (off_diag - off_diag_min) / (off_diag_max - off_diag_min)
    
    loss = on_diag.sum(-1) + lambda_ * off_diag.sum((-2, -1)) + margin
    
    return loss / z_a.size(-1)


def forward(sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
    embeddings = [sentence_feature['sentence_embedding'] for sentence_feature in sentence_features ]
    output = score_transformation( barlow_twins_loss(embeddings[0], embeddings[1]) )
    #print(output, '\n', labels)
    return loss_fct( output, labels.view(-1) )

# ----------------------------------------------------------------------- #





with open("embeddings.txt", "rb") as file:
    embeddings = pickle.load(file)
embeddings = [ {'sentence_embedding': Tensor(embed)} for embed in embeddings ]

labels = Tensor( [0.5200, 0.1600, 0.9600, 0.4000, 0.8400, 0.1600, 0.2800, 0.9000, 0.7200, 0.2000, 0.4800, 0.5600, 0.6800, 0.0000, 0.4800, 0.3600] )



print( forward_single( embeddings, labels ) )
start = time.perf_counter()
forward_single( embeddings, labels )
end = time.perf_counter()
print(f'Time: {end - start:.4}s')


# print(embeddings[0]['sentence_embedding'][:].size())
# print( embeddings[0]['sentence_embedding'][:, :, None].size() )
# print( embeddings[1]['sentence_embedding'][:, None, :].size() )
# print( torch.bmm( embeddings[0]['sentence_embedding'][:, :, None], embeddings[1]['sentence_embedding'][:, None, :] ).size() )
# c = torch.bmm( embeddings[0]['sentence_embedding'][:, :, None], embeddings[1]['sentence_embedding'][:, None, :] )
# print( "c:", c.size() )
# print( torch.diagonal( (1.0 - c), dim1=-2, dim2=-1 ).pow(2).size() )
# on_diag = torch.diagonal( (1.0 - c), dim1=-2, dim2=-1 ).pow(2)
# print( "on_diag:", on_diag.size() )
# off_diag = ( (1 - torch.eye(c.size(-1)) ) * c ).pow(2)
# print( "off_diag:", off_diag.size() )
# print( "on_diag min:", ( on_diag - on_diag.min(1).values[:, None]).size() )
# print( "off_diag min:", ( off_diag - (off_diag.min(2).values).min(1).values[:, None, None] ).size() )


print( forward( embeddings, labels ) )
start = time.perf_counter()
forward( embeddings, labels )
end = time.perf_counter()
print(f'Time: {end - start:.4}s')



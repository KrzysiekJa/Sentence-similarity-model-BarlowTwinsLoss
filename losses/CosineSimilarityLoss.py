import torch
from torch import nn, Tensor
from typing import Iterable


class CosineSimilarityLoss(nn.Module):

    def __init__(
            self,
            loss_fct=nn.MSELoss(), 
            score_transformation=nn.Identity()
            ):
        super(CosineSimilarityLoss, self).__init__()
        self.loss_fct = loss_fct
        self.score_transformation = score_transformation
        self.cos_sim = torch.cosine_similarity()


    def forward(self, input: Iterable, label: Tensor):
        embedding_1 = torch.stack( [inp[0] for inp in input] )
        embedding_2 = torch.stack( [inp[1] for inp in input] )
        output = self.score_transformation( self.cos_sim(embedding_1, embedding_2) )
        return self.loss_fct( output, label.squeeze() )

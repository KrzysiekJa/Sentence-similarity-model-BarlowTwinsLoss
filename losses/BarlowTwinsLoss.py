import torch
from torch import nn, Tensor
from typing import Iterable, Dict
import os, sys
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__) ), '..') )
# ugly code to achieve sentence_transformer module
from sentence_transformer import SentenceTransformer


class BarlowTwinsLoss(nn.Module):
    """
    On basis of: https://arxiv.org/pdf/2103.03230.pdf

    BarlowTwinsLoss expects, that the InputExamples consists of two texts and a float label.
    
    It computes the vectors u = model(input_text[0]) and v = model(input_text[1]) and measures the Barlow Twins loss function for the two.
    By default, it minimizes the following loss: ||input_label - score_transformation(barlow_twins_loss(u,v))||_2.
    
    :param model: SentenceTranformer model
    :param lambda_: float
    :param margin: float
    :param loss_fct: Which pytorch loss function should be used to compare the cosine_similartiy(u,v) with the input_label? By default, MSE:  ||input_label - barlow_twins_loss(u,v)||_2
    :param score_transformation: The score_transformation function is applied on top of barlow_twins_loss. By default, the identify function is used (i.e. no change).
    
    Example::
    
            from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
    
            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
                InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
            train_dataset = SentencesDataset(train_examples, model)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
            train_loss = losses.BarlowTwinsLoss(model=model)
    
    """
    def __init__(
            self, 
            model: SentenceTransformer, 
            lambda_: float = 1e-2,
            margin: float = 1e-12, 
            loss_fct=nn.MSELoss(), 
            score_transformation=nn.Identity() 
            ):
        super(BarlowTwinsLoss, self).__init__()
        self.model = model
        self.lambda_ = lambda_
        self.margin = margin
        self.loss_fct = loss_fct
        self.score_transformation = score_transformation
    
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [ self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features ]
        output = self.score_transformation( self.barlow_twins_loss(embeddings[0], embeddings[1]) )
        return self.loss_fct( output, labels.view(-1) )
    
    def barlow_twins_loss(self, z_a: Tensor, z_b: Tensor) -> Tensor:
        """Compute the loss.
        Args:
            z_a: Batch of tensor vectors of shape (batch_size, embedding_dim)
            z_b: Batch of tensor vectors of shape (batch_size, embedding_dim)
        Returns:
            loss
        """
        assert z_a.size() == z_b.size(), "Embeddings error: different dimensions."
        
        # cross-correlation matrix
        c = torch.bmm( z_a[:, :, None], z_b[:, None, :] ).type(torch.float32) # batch matrix-matrix product
        on_diag  = torch.diagonal( (1.0 - c), dim1=-2, dim2=-1 ).type(torch.float32).pow(2)
        off_diag = ( ( 1.0 - torch.eye(c.size(-1), device=c.device).type(torch.float32) ) * c ).pow(2)

        # normalization: min-max scaling
        on_diag_min  = on_diag.min(1).values[:, None]
        on_diag_max  = on_diag.max(1).values[:, None]
        off_diag_min = (off_diag.min(2).values).min(1).values[:, None, None]
        off_diag_max = (off_diag.max(2).values).max(1).values[:, None, None]
        on_diag  = (on_diag  - on_diag_min)  / (on_diag_max  - on_diag_min)
        off_diag = (off_diag - off_diag_min) / (off_diag_max - off_diag_min)

        # 1D loss Tensor
        loss = on_diag.sum(-1) + self.lambda_ * off_diag.sum((-2, -1)) + self.margin
        
        return loss / z_a.size(-1)

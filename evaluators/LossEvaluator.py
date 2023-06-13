import os
import csv
import logging
from tqdm.autonotebook import trange

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction
from sentence_transformers.readers import InputExample

logger = logging.getLogger(__name__)


class LossEvaluator(SentenceEvaluator):
    # On basis of: https://github.com/UKPLab/sentence-transformers/issues/336
    
    def __init__(self, val_samples, loss_model: nn.Module = None, name: str = '', log_dir: str = None, show_progress_bar: bool = False, write_csv: bool = True, batch_size: int = 16):

        """
        Evaluate a model based on the loss function.
        The returned score is loss value.
        The results are written in a CSV and Tensorboard logs.
        :param val_samples: List[InputExample]
        :param loss_model: loss module object
        :param name: Name for the output
        :param log_dir: path for tensorboard logs 
        :param show_progress_bar: If true, prints a progress bar
        :param write_csv: Write results to a CSV file
        :param batch_size: size of data batches
        """

        self.loader = DataLoader(val_samples, shuffle=True, batch_size=batch_size)
        self.write_csv = write_csv
        self.logs_writer = SummaryWriter(log_dir=log_dir)
        self.name = name
        self.loss_model = loss_model
        self.batch_size = batch_size
        self.val_samples = val_samples
        
        # move model to gpu:  lidija-jovanovska
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        loss_model.to(self.device)

        if show_progress_bar is None:
            show_progress_bar = (
                    logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "loss_evaluation" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "loss"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        global run
        self.loss_model.eval()

        loss_value = 0
        self.loader.collate_fn = model.smart_batching_collate
        num_batches = len(self.loader)
        data_iterator = iter(self.loader)

        with torch.no_grad():
            for _ in trange(num_batches, desc="Iteration", smoothing=0.05, disable=not self.show_progress_bar):
                sentence_features, labels = next(data_iterator)
                for i in range(len(sentence_features)):
                    for key, value in sentence_features[i].items():
                        sentence_features[i][key] = sentence_features[i][key].to(self.device)
                labels = labels.to(self.device)
                loss_value += self.loss_model(sentence_features, labels).item()

        final_loss = loss_value / num_batches

        if output_path is not None and self.write_csv:

            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)

            with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, final_loss])

            # ...log the running loss
            self.logs_writer.add_scalar('val_loss',
                                        final_loss,
                                        steps)

        self.loss_model.zero_grad()
        self.loss_model.train()
        
        run["train/val_loss"].append(final_loss)

        if self.val_samples:
            similarity_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                self.val_samples, batch_size=self.batch_size, main_similarity=SimilarityFunction.COSINE
            )
            evaluation_accuracy = similarity_evaluator(model)
            run["train/val_accuracy"].append(evaluation_accuracy)

        return final_loss
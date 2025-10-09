from time import sleep

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from typing import Dict, Tuple, Any

from navsim.agents.abstract_agent import AbstractAgent
import glob
import os
import subprocess
import shutil
import json

class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self.checkpoint_file=None

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch

        prediction = self.agent.forward(features, targets)
        loss_dict = self.agent.compute_loss(features, targets, prediction)

        if type(loss_dict) is dict:
            # Detach values that are not the main loss to avoid retaining the graph
            # via logging hooks across steps.
            for key, value in loss_dict.items():
                log_val = value
                if key != "loss" and isinstance(value, torch.Tensor):
                    log_val = value.detach()
                self.log(
                    f"{logging_prefix}/" + key,
                    log_val,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    sync_dist=True,
                )
            return loss_dict["loss"]
        else:
            return loss_dict

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        if 'GaussianFusion' in self.agent.name() or 'Pad' in self.agent.name():
            features, targets = batch
            
            predictions = self.agent.forward(features, targets)

            trajectory = predictions.get("trajectory", None)
            if trajectory is not None:

                final_score, best_score, proposal_scores = self.agent.compute_score(targets, trajectory[:, None])
                mean_score = proposal_scores.mean()
                pdm_score = predictions["pdm_score"]
                pdm_score = pdm_score[torch.arange(len(pdm_score)), torch.argmax(pdm_score, dim=1)]
                score_error=torch.abs(pdm_score - proposal_scores).mean()
            else:
                final_score = 0.0
                best_score = 0.0
                mean_score = 0.0
                score_error = 0.0
            logging_prefix="val"
            self.log(f"{logging_prefix}/score", final_score, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/best_score", best_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/mean_score", mean_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/score_error", score_error, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            return final_score
        else:
            return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

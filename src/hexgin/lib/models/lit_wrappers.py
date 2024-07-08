"""Contains PyTorch Lightning wrappers for models - a common link prediction model that
has any GNN architecture instide."""

import torch as th
import torch_geometric.data as tgdata
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Dict, Any

import torchmetrics as tm


class LinkPredLitWrapper(pl.LightningModule):
    def __init__(
        self,
        model: th.nn.Module,
        name: str,
        target_relation: str,
        learning_rate: float = 0.01,
        batch_size: int = 256,
        params: Dict[str, Any] = {},
    ):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.params = params
        self.model_name = name
        self.model = model
        self.batch_size = batch_size
        self.target_relation = target_relation
        self.learning_rate = learning_rate

    def forward(self, data: tgdata.HeteroData, use_labeled_edges: bool = True):
        return self.model(data, use_labeled_edges=use_labeled_edges).squeeze()

    def _step(self, batch, phase: str = "training", use_labeled_edges=True):
        pred = self(batch, use_labeled_edges=use_labeled_edges)
        y = batch[self.target_relation].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, y)
        f1_score = tm.functional.f1_score(pred, y, task="binary", threshold=0.5)
        self.log(f"{phase}_loss", loss.item(), on_step=True, batch_size=self.batch_size)
        self.log(
            f"{phase}_f1",
            f1_score.item(),
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, *args, **kwargs):
        pred = self(batch)
        y = batch[self.target_relation].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, y)
        self.log("test_loss", loss, on_step=True, batch_size=self.batch_size)
        f1_score = tm.functional.f1_score(pred, y, task="binary", threshold=0.5)
        prec = tm.functional.precision(pred, y, task="binary", threshold=0.5)
        rec = tm.functional.recall(pred, y, task="binary", threshold=0.5)
        self.log(
            "test_f1",
            f1_score,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "test_precision",
            prec,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "test_recall",
            rec,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        return loss

    def configure_optimizers(self):
        return th.optim.Adam(self.parameters(), lr=self.learning_rate)

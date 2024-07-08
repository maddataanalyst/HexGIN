"""A set of common trianing functions used across all models."""

from typing import Tuple

import pandas as pd
import pytorch_lightning as pl
import torch as th
import torch_geometric as tg
import logging as logging
import torchmetrics as tm
from torch_geometric import loader as tgloader, data as tgdata

from hexgin import consts as cc, utils as util
from hexgin.lib.models import lit_wrappers as litw


log = logging.getLogger(__name__)


def train_model(
    model: litw.LinkPredLitWrapper,
    train_loader: tgloader.LinkNeighborLoader,
    val_loader: tgloader.LinkNeighborLoader,
    max_epochs: int = 20,
    experiment_name: str = "default",
    **kwargs,
) -> pl.Trainer:
    """A generic Pytorch Lightning training function.

    Parameters
    ----------
    model : litw.LinkPredLitWrapper
        Link prediction model - a Pytorch Lightning wrapper for any of the
        architectures in the experiment (e.g. HexGIN, GIN, etc.)
    train_loader : tgloader.LinkNeighborLoader
        Graph trianing loader.
    val_loader : tgloader.LinkNeighborLoader
        Graph validation loader.
    max_epochs : int, optional
        Maximal number of epochs, by default 20
    experiment_name : str, optional
        Name of the experiment, by default "default"

    Returns
    -------
    pl.Trainer
        Prepared Pytorch Lightning trainer for the model.
    """
    loggers = [
        pl.loggers.MLFlowLogger(
            tracking_uri="file:./mlruns", experiment_name=experiment_name
        ),
        pl.loggers.TensorBoardLogger(save_dir="./"),
    ]

    early_stop = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_f1", patience=2, mode="max"
    )
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor="val_f1", mode="max", save_top_k=1
    )
    trainer = pl.Trainer(
        check_val_every_n_epoch=3,
        max_epochs=max_epochs,
        logger=loggers,
        enable_checkpointing=True,
        callbacks=[early_stop, checkpoint],
        accelerator="cpu",
        log_every_n_steps=5,
        **kwargs.get("trainer_kwargs", {}),
    )
    pl.seed_everything(123)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return trainer


def test_model(
    model: litw.LinkPredLitWrapper, trainer: pl.Trainer, test_data: tgdata.HeteroData
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Performs Pytorch Lightning model testing.

    Parameters
    ----------
    model : litw.LinkPredLitWrapper
        Link prediction model, with any architecture inside (e.g. HexGIN, GIN, etc.).
    trainer : pl.Trainer
        Trianer for the model.
    test_data : tgdata.HeteroData
        Hetero graph test data.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Training results: classification report and metrics.
    """

    test_loader = tgdata.DataLoader([test_data], batch_size=1, shuffle=False)

    trainer.test(model, test_loader, ckpt_path="best")

    with th.no_grad():
        model.eval()
        yhat_test = model(test_data, use_labeled_edges=True)
        y = test_data[cc.TARGET_RELATION].edge_label

    yhat_test_prob = th.sigmoid(yhat_test)

    clf_report_df = util.sklearn_classification_report_to_pandas(
        y, yhat_test_prob > 0.5
    )
    f1 = tm.functional.f1_score(yhat_test_prob, y, task="binary").item()
    prec = tm.functional.precision(yhat_test_prob, y, task="binary").item()
    recall = tm.functional.recall(yhat_test_prob, y, task="binary").item()
    rocauc = tm.functional.auroc(yhat_test_prob, y.to(th.long), task="binary").item()

    metrics_df = pd.DataFrame(
        {
            "f1": [f1],
            "precision": [prec],
            "recall": [recall],
            "rocauc": [rocauc],
            "model": [model.model_name],
        }
    )

    return clf_report_df, metrics_df


def train_and_test_model(
    model: litw.LinkPredLitWrapper,
    train_data: tgdata.HeteroData,
    val_data: tgdata.HeteroData,
    test_data: tgdata.HeteroData,
    training_params: dict,
    experiment_name: str = "train_and_test",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Performs both - trainig model on full data and testing it.

    Parameters
    ----------
    model : litw.LinkPredLitWrapper
        Link prediction model, with any architecture inside (e.g. HexGIN, GIN, etc.).
    train_data : tgdata.HeteroData
        Train graph.
    val_data : tgdata.HeteroData
        Validation graph.
    test_data : tgdata.HeteroData
        Test graph.
    training_params : dict
        Training parameters.
    experiment_name : str, optional
        Experiment name for MlFlow, by default "train_and_test"

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Training metrics: classification report and metrics data frames.
    """
    th.use_deterministic_algorithms(training_params["deterministic"])
    tg.seed_everything(training_params["seed"])
    train_loader = tg.loader.LinkNeighborLoader(
        data=train_data,
        num_neighbors=training_params["num_neighbors"],
        edge_label_index=(
            cc.TARGET_RELATION,
            train_data[cc.TARGET_RELATION].edge_label_index,
        ),
        edge_label=train_data[cc.TARGET_RELATION].edge_label,
        directed=False,
        neg_sampling_ratio=training_params["neg_sampling_ratio"],
        batch_size=training_params["batch_size"],
        shuffle=training_params["shuffle"],
    )
    val_loader = tgdata.DataLoader([val_data], batch_size=1, shuffle=False)

    trainer = train_model(
        model,
        train_loader,
        val_loader,
        max_epochs=training_params["max_epochs"],
        experiment_name=experiment_name,
        trainer_kwargs={"deterministic": training_params["deterministic"]},
    )
    clf_report_df, metrics_df = test_model(model, trainer, test_data)
    return clf_report_df, metrics_df


def cross_val_models(
    models: Tuple[litw.LinkPredLitWrapper, ...],
    train_data: tgdata.HeteroData,
    cross_val_params: dict,
    training_params: dict,
) -> pd.DataFrame:
    """Performs graph cross-validation for a set of models.

    Parameters
    ----------
    models : Tuple[litw.LinkPredLitWrapper, ...]
        Tuple of link prediction models to cross validate.
    train_data : tgdata.HeteroData
        Trianing graph.
    cross_val_params : dict
        Cross-validation parameters.
    training_params : dict
        Training parameters.

    Returns
    -------
    pd.DataFrame
        Cross validation metrics results.
    """
    train_data_no_labels = train_data.clone()
    del train_data_no_labels[cc.TARGET_RELATION].edge_label
    del train_data_no_labels[cc.TARGET_RELATION].edge_label_index
    tg.seed_everything(training_params["seed"])
    splitter = tg.transforms.RandomLinkSplit(
        num_val=cross_val_params[cc.CFG_NUM_VAL],
        num_test=cross_val_params[cc.CFG_NUM_TEST],
        neg_sampling_ratio=cross_val_params[cc.CFG_NEG_SAMPLING_RATIO],
        add_negative_train_samples=cross_val_params[cc.CFG_ADD_NEGATIVE_TRAIN_SAMPLES],
        disjoint_train_ratio=cross_val_params[cc.CFG_DISJOINT_TRAIN_RATIO],
        edge_types=cc.TARGET_RELATION,
    )
    nfolds = cross_val_params[cc.CFG_NUM_FOLDS]
    experiment_name = cross_val_params[cc.CFG_EXPERIMENT_NAME]
    cross_val_metrics = []
    for i in range(nfolds):
        tg.seed_everything(i)
        cv_train, _, cv_test = splitter(train_data_no_labels)
        for model in models:
            log.info(f"Startig crossval fold {i} for model {model.model_name}")
            _, model_metrics = train_and_test_model(
                model,
                cv_train,
                cv_test,
                cv_test,
                training_params,
                experiment_name=experiment_name,
            )
            model_metrics["fold"] = i
            cross_val_metrics.append(model_metrics)
    all_metrics = pd.concat(cross_val_metrics)
    return all_metrics

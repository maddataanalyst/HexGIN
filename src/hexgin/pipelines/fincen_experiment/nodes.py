"""The module contains code of main experimentation methods and procedures used in the HexGIN model."""

import torch_geometric as tg
import torch_geometric.data as tgdata
import pandas as pd
import numpy as np
from typing import Tuple

from sklearn.preprocessing import LabelEncoder
from scipy.stats import bootstrap

import hexgin.pipelines.fincen_experiment.fincen_model_utils as fmu
import hexgin.lib.training_commons as train_cmn
import hexgin.lib.models.lit_wrappers as litw
import hexgin.consts as cc


def prepare_mlp(
    dim_country: int, dim_entity: int, dim_filing: int, model_params: dict, seed: int
) -> litw.LinkPredLitWrapper:
    """A generic function for preparing an MLP model for the link prediction task.

    Parameters
    ----------
    dim_country : int
        The number of countries.
    dim_entity : int
        The number of entities.
    dim_filing : int
        The number of filings.
    model_params : dict
        Dictionary with model parameters.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    litw.LinkPredLitWrapper
        A link prediction model wrapped in a LitWrapper.
    """
    tg.seed_everything(seed)
    mlp = fmu.build_mlp(dim_entity, dim_country, dim_filing, model_params["MLP_params"])
    mlp_lit = litw.LinkPredLitWrapper(
        mlp, "MLP", cc.TARGET_RELATION, params=model_params["MLP_params"]
    )
    return mlp_lit


def prepare_sage(
    dim_country: int,
    dim_entity: int,
    dim_filing: int,
    model_params: dict,
    train_data,
    seed: int,
) -> litw.LinkPredLitWrapper:
    """Builds a Graph SAGE model, using the specified parameters.

    Parameters
    ----------
    dim_country : int
        The number of countries.
    dim_entity : int
        The number of entities.
    dim_filing : int
        The number of filings.
    model_params : dict
        Dictionary with model parameters.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    litw.LinkPredLitWrapper
        A link predictor model with a Graph SAGE architecture.
    """
    tg.seed_everything(seed)
    sage = fmu.build_graph_sage_model(
        train_data, dim_entity, dim_country, dim_filing, model_params["SAGE_params"]
    )
    sage_lit = litw.LinkPredLitWrapper(
        sage, "SAGE", cc.TARGET_RELATION, params=model_params["SAGE_params"]
    )
    return sage_lit


def prepare_hexgin(
    dim_country: int, dim_entity: int, dim_filing: int, model_params: dict, seed: int
) -> litw.LinkPredLitWrapper:
    """Prepares a HexGIN model for the link prediction task.

    dim_country : int
        The number of countries.
    dim_entity : int
        The number of entities.
    dim_filing : int
        The number of filings.
    model_params : dict
        Dictionary with model parameters.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    litw.LinkPredLitWrapper
        A link predictor model with a HexGIN model architecture.
    """
    tg.seed_everything(seed)
    hexgin = fmu.build_hexgin_net(
        dim_entity, dim_country, dim_filing, model_params["HexGIN_params"]
    )
    hexgin_lit = litw.LinkPredLitWrapper(
        hexgin,
        "HexGIN",
        cc.TARGET_RELATION,
        learning_rate=0.001,
        params=model_params["HexGIN_params"],
    )
    return hexgin_lit


def bootstrap_metrics(metrics_data: pd.DataFrame, n_samples=1000) -> pd.DataFrame:
    """Performs a bootstrap procedure on the metrics data and calculates boostrapped confidence intervals.

    Parameters
    ----------
    metrics_data : pd.DataFrame
        Original metrics data.
    n_samples : int, optional
        Number of resamples, by default 1000

    Returns
    -------
    pd.DataFrame
        Bootstraped metrics data.
    """
    metrics = ["f1", "precision", "recall", "rocauc"]
    bootstraped_metrics = []
    for model in metrics_data.model.unique():
        for metric in metrics:
            vals = metrics_data.loc[(metrics_data.model == model), metric]
            vals_boot = bootstrap(
                (vals.values,),
                np.mean,
                confidence_level=0.95,
                random_state=123,
                n_resamples=1000,
                method="percentile",
            )
            bootstraped_metrics.append(
                {
                    "model": model,
                    "metric": metric,
                    "CI low": vals_boot.confidence_interval.low,
                    "CI high": vals_boot.confidence_interval.high,
                }
            )

    bootstraped_metrics = pd.DataFrame(bootstraped_metrics)
    return bootstraped_metrics


def cross_validate_models(
    train_data: tgdata.HeteroData,
    entity_encoder: LabelEncoder,
    filing_encoder: LabelEncoder,
    country_encoder: LabelEncoder,
    model_params: dict,
    crossval_params: dict,
    training_params: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Performs a full cross-validation of the HexGIN, SAGE, and MLP models and returns the results.

    Parameters
    ----------
    train_data : tgdata.HeteroData
        Training data - a heterogeneous graph.
    entity_encoder : LabelEncoder
        Encoder for entities.
    filing_encoder : LabelEncoder
        Encoder for filings.
    country_encoder : LabelEncoder
        Encoder for countries.
    model_params : dict
        Model parameters.
    crossval_params : dict
        Cross-validation parameters.
    training_params : dict
        Training parameters.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Cross-validation metrics, melted metrics, and bootstraped metrics.
    """
    dim_entity = len(entity_encoder.classes_)
    dim_country = len(country_encoder.classes_)
    dim_filing = len(filing_encoder.classes_)

    hexgin_lit = prepare_hexgin(
        dim_country, dim_entity, dim_filing, model_params, training_params["seed"]
    )
    sage_lit = prepare_sage(
        dim_country,
        dim_entity,
        dim_filing,
        model_params,
        train_data,
        training_params["seed"],
    )
    mlp_lit = prepare_mlp(
        dim_country, dim_entity, dim_filing, model_params, training_params["seed"]
    )

    models = [hexgin_lit, sage_lit, mlp_lit]
    cross_val_metrics = train_cmn.cross_val_models(
        models, train_data, crossval_params, training_params
    )
    melted_metrics = (
        cross_val_metrics.melt(
            id_vars=["model", "fold"],
            value_vars=["precision", "f1", "recall", "rocauc"],
            var_name="metric",
        )
        .groupby(["model", "metric"])
        .agg({"value": ["mean", "median", "std"]})
        .T
    )
    bootstraped_metrics = bootstrap_metrics(cross_val_metrics)
    return cross_val_metrics, melted_metrics, bootstraped_metrics


def generic_train_model(
    train_data: tgdata.HeteroData,
    val_data: tgdata.HeteroData,
    test_data: tgdata.HeteroData,
    entity_encoder: LabelEncoder,
    filing_encoder: LabelEncoder,
    country_encoder: LabelEncoder,
    model_builder_func: callable,
    model_params: dict,
    training_params: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A generic function for training a model and testing it on the test data.

    Parameters
    ----------
    train_data : tgdata.HeteroData
        Training heterogeneous graph.
    val_data : tgdata.HeteroData
        Validation heterogeneous graph.
    test_data : tgdata.HeteroData
        Testing heterogeneous graph.
    entity_encoder : LabelEncoder
        Encoder for entities.
    filing_encoder : LabelEncoder
        Encoder for filings.
    country_encoder : LabelEncoder
        Encoder for countries.
    model_builder_func : callable
        A factory function that builds a model.
    model_params : dict
        Model parameters.
    training_params : dict
        Training params.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple of classification report and test metrics.
    """
    dim_entity = len(entity_encoder.classes_)
    dim_country = len(country_encoder.classes_)
    dim_filing = len(filing_encoder.classes_)
    tg.seed_everything(training_params["seed"])

    lit_model = model_builder_func(
        dim_country, dim_entity, dim_filing, model_params, training_params["seed"]
    )

    clf_report, test_metrics = train_cmn.train_and_test_model(
        lit_model, train_data, val_data, test_data, training_params
    )
    return clf_report, test_metrics


def train_hexgin(
    train_data: tgdata.HeteroData,
    val_data: tgdata.HeteroData,
    test_data: tgdata.HeteroData,
    entity_encoder: LabelEncoder,
    filing_encoder: LabelEncoder,
    country_encoder: LabelEncoder,
    model_params: dict,
    training_params: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A specific function for training the HexGIN model.

    Parameters
    ----------
    train_data : tgdata.HeteroData
        Training heterogeneous graph.
    val_data : tgdata.HeteroData
        Validation heterogeneous graph.
    test_data : tgdata.HeteroData
        Testing heterogeneous graph.
    entity_encoder : LabelEncoder
        Encoder for entities.
    filing_encoder : LabelEncoder
        Encoder for filings.
    country_encoder : LabelEncoder
        Encoder for countries.
    model_builder_func : callable
        A factory function that builds a model.
    model_params : dict
        Model parameters.
    training_params : dict
        Training params._

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple of classification report and test metrics.
    """
    return generic_train_model(
        train_data,
        val_data,
        test_data,
        entity_encoder,
        filing_encoder,
        country_encoder,
        prepare_hexgin,
        model_params,
        training_params,
    )


def train_mlp(
    train_data: tgdata.HeteroData,
    val_data: tgdata.HeteroData,
    test_data: tgdata.HeteroData,
    entity_encoder: LabelEncoder,
    filing_encoder: LabelEncoder,
    country_encoder: LabelEncoder,
    model_params: dict,
    training_params: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A specific function for training the Multi-layer perceptron model.

    Parameters
    ----------
    train_data : tgdata.HeteroData
        Training heterogeneous graph.
    val_data : tgdata.HeteroData
        Validation heterogeneous graph.
    test_data : tgdata.HeteroData
        Testing heterogeneous graph.
    entity_encoder : LabelEncoder
        Encoder for entities.
    filing_encoder : LabelEncoder
        Encoder for filings.
    country_encoder : LabelEncoder
        Encoder for countries.
    model_builder_func : callable
        A factory function that builds a model.
    model_params : dict
        Model parameters.
    training_params : dict
        Training params._

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple of classification report and test metrics.
    """
    return generic_train_model(
        train_data,
        val_data,
        test_data,
        entity_encoder,
        filing_encoder,
        country_encoder,
        prepare_mlp,
        model_params,
        training_params,
    )


def train_sage(
    train_data: tgdata.HeteroData,
    val_data: tgdata.HeteroData,
    test_data: tgdata.HeteroData,
    entity_encoder: LabelEncoder,
    filing_encoder: LabelEncoder,
    country_encoder: LabelEncoder,
    model_params: dict,
    training_params: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A specific function for training the GraphSAGE model.

    Parameters
    ----------
    train_data : tgdata.HeteroData
        Training heterogeneous graph.
    val_data : tgdata.HeteroData
        Validation heterogeneous graph.
    test_data : tgdata.HeteroData
        Testing heterogeneous graph.
    entity_encoder : LabelEncoder
        Encoder for entities.
    filing_encoder : LabelEncoder
        Encoder for filings.
    country_encoder : LabelEncoder
        Encoder for countries.
    model_builder_func : callable
        A factory function that builds a model.
    model_params : dict
        Model parameters.
    training_params : dict
        Training params._

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple of classification report and test metrics.
    """

    # Because SAGE is constructed using tg.nn.to_hetero(...) which requires graph metadata, train_graph needs
    # to be passed as argument

    def builder_f(dim_c, dim_e, dim_f, params, seed):
        return prepare_sage(
            dim_c, dim_e, dim_f, params, train_data=train_data, seed=seed
        )

    return generic_train_model(
        train_data,
        val_data,
        test_data,
        entity_encoder,
        filing_encoder,
        country_encoder,
        builder_f,
        model_params,
        training_params,
    )

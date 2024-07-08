"""Module contains main processing functions, used in the data preparation pipeline.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler

log = logging.getLogger(__name__)


def find_common_entities(
    entities: pd.DataFrame,
    originators_beneficiaries: pd.DataFrame,
    concerned: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find entities that are present in either originators_beneficiaries or concerned. Keep only
    those entities.

    Parameters
    ----------
    entities: pd.DataFrame
        The entities dataframe
    filings: pd.DataFrame
        The filings dataframe
    originators_beneficiaries: pd.DataFrame
        The originators_beneficiaries dataframe
    concerned: pd.DataFrame
        The concerned dataframe

    Returns
    -------
    pd.DataFrame
        Find entities subset
    """
    either_present = (
        entities.eid.isin(originators_beneficiaries.oid)
        | entities.eid.isin(originators_beneficiaries.bid)
        | (entities.eid.isin(concerned.cid))
    )

    entities_sset = entities[either_present]
    log.info(f"Original entities: {len(entities)}")
    log.info(f"Number of common entities: {len(entities_sset)}")

    return entities_sset


def prepare_encoders_and_scaler(
    entities: pd.DataFrame, filings: pd.DataFrame
) -> Tuple[LabelEncoder, LabelEncoder, LabelEncoder, StandardScaler]:
    """
    Prepares encoders and scaler.

    Parameters
    ----------
    entities: pd.DataFrame
        The entities dataframe
    filings: pd.DataFrame
        The filings dataframe

    Returns
    -------
    Tuple[LabelEncoder, LabelEncoder, LabelEncoder, StandardScaler]
        The encoders and scaler
    """
    entity_encoder = LabelEncoder()
    filing_encoder = LabelEncoder()
    country_encoder = LabelEncoder()
    amnt_scaler = StandardScaler()

    entity_encoder.fit(entities.eid)
    filing_encoder.fit(filings.fid)
    country_encoder.fit(entities.country)
    amnt_scaler.fit(filings.famt.values.reshape(-1, 1))

    return entity_encoder, filing_encoder, country_encoder, amnt_scaler


def prepare_matrices(
    entities: pd.DataFrame,
    concerned: pd.DataFrame,
    filings: pd.DataFrame,
    originators_beneficiaries: pd.DataFrame,
    entity_encoder: LabelEncoder,
    filing_encoder: LabelEncoder,
    country_encoder: LabelEncoder,
) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    Prepares matrices for the model by applying encoders to the dataframes.

    Parameters
    ----------
    entities: pd.DataFrame
        The entities dataframe
    concerned: pd.DataFrame
        The concerned dataframe
    filings: pd.DataFrame
        The filings dataframe
    originators_beneficiaries: pd.DataFrame
        The originators_beneficiaries dataframe
    entity_encoder: LabelEncoder
        The entity_encoder
    filing_encoder: LabelEncoder
        The filing_encoder
    country_encoder: LabelEncoder
        The country_encoder

    Returns
    -------
    Tuple[np.array, np.array, np.array, np.array, np.array]
        The matrices:
        1. Filings id matrix
        2. Originator to filing matrix
        3. Filing to beneficiary matrix
        4. Filing to concerned matrix
        5. Entity ids matrix
    """
    filings = filing_encoder.transform(filings.fid.values).reshape((1, -1)).T
    originator2filing = np.vstack(
        (
            entity_encoder.transform(originators_beneficiaries.oid),
            filing_encoder.transform(originators_beneficiaries.fid),
        )
    )
    filing2beneficiary = np.vstack(
        (
            filing_encoder.transform(originators_beneficiaries.fid),
            entity_encoder.transform(originators_beneficiaries.bid),
        )
    )
    filing2concerned = np.vstack(
        (
            filing_encoder.transform(concerned.fid),
            entity_encoder.transform(concerned.cid),
        )
    )

    entity_ids = np.vstack(
        (
            entity_encoder.transform(entities.eid),
            country_encoder.transform(entities.country),
        )
    ).T

    return filings, originator2filing, filing2beneficiary, filing2concerned, entity_ids

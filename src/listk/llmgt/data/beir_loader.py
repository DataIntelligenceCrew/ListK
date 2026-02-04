"""BEIR dataset downloading and loading utilities.

This module provides functions to download BEIR benchmark datasets
and load them into structured data models for use in the pipeline.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import DatasetInfo, Document, Query, Qrel

logger = logging.getLogger(__name__)

# Supported BEIR datasets for this project
SUPPORTED_DATASETS: set[str] = {"scifact", "scidocs"}

# BEIR dataset URLs (from the official BEIR repository)
BEIR_DATASET_URLS: dict[str, str] = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "scidocs": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip",
}


class BeirDataset:
    """A loaded BEIR dataset with corpus, queries, and relevance judgments.

    Attributes
    ----------
    name : str
        The dataset name.
    corpus : dict[str, Document]
        Mapping from doc_id to Document objects.
    queries : dict[str, Query]
        Mapping from query_id to Query objects.
    qrels : dict[str, dict[str, int]]
        Nested dict: qrels[query_id][doc_id] = relevance_score.
    info : DatasetInfo
        Metadata about the dataset.
    """

    def __init__(
        self,
        name: str,
        corpus: dict[str, Document],
        queries: dict[str, Query],
        qrels: dict[str, dict[str, int]],
        info: DatasetInfo,
    ) -> None:
        """Initialize a BeirDataset.

        Parameters
        ----------
        name : str
            Dataset name.
        corpus : dict[str, Document]
            Document mapping.
        queries : dict[str, Query]
            Query mapping.
        qrels : dict[str, dict[str, int]]
            Relevance judgments.
        info : DatasetInfo
            Dataset metadata.
        """
        self.name: str = name
        self.corpus: dict[str, Document] = corpus
        self.queries: dict[str, Query] = queries
        self.qrels: dict[str, dict[str, int]] = qrels
        self.info: DatasetInfo = info

    def get_relevant_docs(self, query_id: str) -> dict[str, int]:
        """Get relevant documents for a query.

        Parameters
        ----------
        query_id : str
            The query identifier.

        Returns
        -------
        dict[str, int]
            Mapping from doc_id to relevance score for the query.
        """
        return self.qrels.get(query_id, {})

    def get_query_ids(self) -> list[str]:
        """Get all query IDs in the dataset.

        Returns
        -------
        list[str]
            List of query identifiers.
        """
        return list(self.queries.keys())

    def get_doc_ids(self) -> list[str]:
        """Get all document IDs in the corpus.

        Returns
        -------
        list[str]
            List of document identifiers.
        """
        return list(self.corpus.keys())


def validate_dataset_name(dataset_name: str) -> None:
    """Validate that the dataset name is supported.

    Parameters
    ----------
    dataset_name : str
        The dataset name to validate.

    Raises
    ------
    ValueError
        If the dataset is not in the supported list.
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' is not supported. "
            f"Supported datasets: {SUPPORTED_DATASETS}"
        )


def get_dataset_path(data_dir: Path, dataset_name: str) -> Path:
    """Get the path where a dataset should be stored.

    Parameters
    ----------
    data_dir : Path
        Base data directory (e.g., ./data).
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    Path
        Path to the dataset directory.
    """
    return data_dir / "raw" / dataset_name


def is_dataset_downloaded(data_dir: Path, dataset_name: str) -> bool:
    """Check if a dataset has already been downloaded.

    Parameters
    ----------
    data_dir : Path
        Base data directory.
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    bool
        True if the dataset exists and appears complete.
    """
    dataset_path: Path = get_dataset_path(data_dir, dataset_name)

    # Check for required files
    required_files: list[str] = ["corpus.jsonl", "queries.jsonl", "metadata.json"]
    qrels_dir: Path = dataset_path / "qrels"

    if not dataset_path.exists():
        return False

    for filename in required_files:
        if not (dataset_path / filename).exists():
            return False

    # Check that qrels directory exists and has at least one file
    if not qrels_dir.exists() or not any(qrels_dir.iterdir()):
        return False

    return True


def download_dataset(
    dataset_name: str,
    data_dir: Path,
    force: bool = False,
) -> Path:
    """Download a BEIR dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to download (e.g., 'scifact').
    data_dir : Path
        Base data directory where datasets are stored.
    force : bool, optional
        If True, re-download even if already exists. Default False.

    Returns
    -------
    Path
        Path to the downloaded dataset directory.

    Raises
    ------
    ValueError
        If the dataset name is not supported.
    """
    validate_dataset_name(dataset_name)

    dataset_path: Path = get_dataset_path(data_dir, dataset_name)

    if is_dataset_downloaded(data_dir, dataset_name) and not force:
        logger.info(f"Dataset '{dataset_name}' already exists at {dataset_path}")
        return dataset_path

    logger.info(f"Downloading dataset '{dataset_name}'...")

    # Import beir here to avoid import errors if not installed
    try:
        from beir import util as beir_util
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError as e:
        raise ImportError(
            "The 'beir' package is required for downloading datasets. "
            "Install it with: pip install beir"
        ) from e

    # Create the raw directory if it doesn't exist
    raw_dir: Path = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download using BEIR utility
    url: str = BEIR_DATASET_URLS[dataset_name]
    download_path: str = beir_util.download_and_unzip(url, str(raw_dir))

    logger.info(f"Dataset downloaded to {download_path}")

    # Load the dataset to get statistics
    corpus, queries, qrels = GenericDataLoader(download_path).load(split="test")

    # Get BEIR version if available
    beir_version: Optional[str] = None
    try:
        import beir
        beir_version = getattr(beir, "__version__", None)
    except Exception:
        pass

    # Create metadata
    info = DatasetInfo(
        name=dataset_name,
        num_documents=len(corpus),
        num_queries=len(queries),
        num_qrels=sum(len(docs) for docs in qrels.values()),
        download_timestamp=datetime.utcnow(),
        beir_version=beir_version,
        source_url=url,
    )

    # Save metadata
    metadata_path: Path = dataset_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(info.model_dump(mode="json"), f, indent=2, default=str)

    logger.info(
        f"Dataset '{dataset_name}' ready: "
        f"{info.num_documents} docs, {info.num_queries} queries, {info.num_qrels} qrels"
    )

    return dataset_path


def load_corpus(dataset_path: Path) -> dict[str, Document]:
    """Load the corpus from a downloaded dataset.

    Parameters
    ----------
    dataset_path : Path
        Path to the dataset directory.

    Returns
    -------
    dict[str, Document]
        Mapping from doc_id to Document objects.
    """
    corpus_file: Path = dataset_path / "corpus.jsonl"
    corpus: dict[str, Document] = {}

    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            doc_data: dict = json.loads(line.strip())
            doc_id: str = doc_data.pop("_id")
            title: str = doc_data.pop("title", "")
            text: str = doc_data.pop("text", "")

            corpus[doc_id] = Document(
                doc_id=doc_id,
                title=title,
                text=text,
                metadata=doc_data,  # Remaining fields as metadata
            )

    return corpus


def load_queries(dataset_path: Path) -> dict[str, Query]:
    """Load queries from a downloaded dataset.

    Parameters
    ----------
    dataset_path : Path
        Path to the dataset directory.

    Returns
    -------
    dict[str, Query]
        Mapping from query_id to Query objects.
    """
    queries_file: Path = dataset_path / "queries.jsonl"
    queries: dict[str, Query] = {}

    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            query_data: dict = json.loads(line.strip())
            query_id: str = query_data.pop("_id")
            text: str = query_data.pop("text", "")

            queries[query_id] = Query(
                query_id=query_id,
                text=text,
                metadata=query_data,  # Remaining fields as metadata
            )

    return queries


def load_qrels(dataset_path: Path, split: str = "test") -> dict[str, dict[str, int]]:
    """Load relevance judgments from a downloaded dataset.

    Parameters
    ----------
    dataset_path : Path
        Path to the dataset directory.
    split : str, optional
        Which split to load ('test', 'dev', 'train'). Default 'test'.

    Returns
    -------
    dict[str, dict[str, int]]
        Nested dict: qrels[query_id][doc_id] = relevance_score.
    """
    qrels_file: Path = dataset_path / "qrels" / f"{split}.tsv"
    qrels: dict[str, dict[str, int]] = {}

    if not qrels_file.exists():
        logger.warning(f"Qrels file not found: {qrels_file}")
        return qrels

    with open(qrels_file, "r", encoding="utf-8") as f:
        # Skip header line
        next(f, None)

        for line in f:
            parts: list[str] = line.strip().split("\t")
            if len(parts) >= 3:
                query_id: str = parts[0]
                doc_id: str = parts[1]
                relevance: int = int(parts[2])

                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = relevance

    return qrels


def load_dataset(
    dataset_name: str,
    data_dir: Path,
    split: str = "test",
    download_if_missing: bool = True,
) -> BeirDataset:
    """Load a BEIR dataset, downloading if necessary.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., 'scifact').
    data_dir : Path
        Base data directory.
    split : str, optional
        Which qrels split to load. Default 'test'.
    download_if_missing : bool, optional
        Whether to download if not found. Default True.

    Returns
    -------
    BeirDataset
        The loaded dataset.

    Raises
    ------
    ValueError
        If dataset is not supported.
    FileNotFoundError
        If dataset is not found and download_if_missing is False.
    """
    validate_dataset_name(dataset_name)

    dataset_path: Path = get_dataset_path(data_dir, dataset_name)

    if not is_dataset_downloaded(data_dir, dataset_name):
        if download_if_missing:
            download_dataset(dataset_name, data_dir)
        else:
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' not found at {dataset_path}. "
                "Run download first or set download_if_missing=True."
            )

    logger.info(f"Loading dataset '{dataset_name}' from {dataset_path}")

    # Load components
    corpus: dict[str, Document] = load_corpus(dataset_path)
    queries: dict[str, Query] = load_queries(dataset_path)
    qrels: dict[str, dict[str, int]] = load_qrels(dataset_path, split)

    # Load metadata
    metadata_path: Path = dataset_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            info = DatasetInfo(**json.load(f))
    else:
        # Create metadata if it doesn't exist
        info = DatasetInfo(
            name=dataset_name,
            num_documents=len(corpus),
            num_queries=len(queries),
            num_qrels=sum(len(docs) for docs in qrels.values()),
        )

    logger.info(
        f"Loaded '{dataset_name}': "
        f"{len(corpus)} docs, {len(queries)} queries, "
        f"{sum(len(d) for d in qrels.values())} qrels"
    )

    return BeirDataset(
        name=dataset_name,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        info=info,
    )


def download_datasets(
    dataset_names: list[str],
    data_dir: Path,
    force: bool = False,
) -> dict[str, Path]:
    """Download multiple BEIR datasets.

    Parameters
    ----------
    dataset_names : list[str]
        List of dataset names to download.
    data_dir : Path
        Base data directory.
    force : bool, optional
        If True, re-download even if exists. Default False.

    Returns
    -------
    dict[str, Path]
        Mapping from dataset name to download path.
    """
    results: dict[str, Path] = {}

    for name in dataset_names:
        try:
            path: Path = download_dataset(name, data_dir, force=force)
            results[name] = path
        except Exception as e:
            logger.error(f"Failed to download '{name}': {e}")
            raise

    return results


def list_available_datasets() -> set[str]:
    """List available datasets that can be downloaded.

    Returns
    -------
    set[str]
        Set of supported dataset names.
    """
    return SUPPORTED_DATASETS.copy()


def get_qrels_as_list(
    qrels: dict[str, dict[str, int]]
) -> list[Qrel]:
    """Convert nested qrels dict to list of Qrel objects.

    Parameters
    ----------
    qrels : dict[str, dict[str, int]]
        Nested qrels dictionary.

    Returns
    -------
    list[Qrel]
        List of Qrel objects.
    """
    result: list[Qrel] = []

    for query_id, doc_rels in qrels.items():
        for doc_id, relevance in doc_rels.items():
            result.append(Qrel(
                query_id=query_id,
                doc_id=doc_id,
                relevance=relevance,
            ))

    return result

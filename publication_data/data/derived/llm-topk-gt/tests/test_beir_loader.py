"""Unit tests for BEIR dataset loader."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.beir_loader import (
    SUPPORTED_DATASETS,
    BeirDataset,
    validate_dataset_name,
    get_dataset_path,
    is_dataset_downloaded,
    load_corpus,
    load_queries,
    load_qrels,
    load_dataset,
    list_available_datasets,
    get_qrels_as_list,
)
from src.data.models import Document, Query, Qrel


class TestValidation:
    """Tests for validation functions."""

    def test_validate_supported_dataset(self) -> None:
        """Test validation passes for supported datasets."""
        # Should not raise
        validate_dataset_name("scifact")
        validate_dataset_name("scidocs")

    def test_validate_unsupported_dataset(self) -> None:
        """Test validation fails for unsupported datasets."""
        with pytest.raises(ValueError, match="not supported"):
            validate_dataset_name("msmarco")

    def test_list_available_datasets(self) -> None:
        """Test listing available datasets."""
        available: set[str] = list_available_datasets()

        assert "scifact" in available
        assert "scidocs" in available
        assert len(available) == len(SUPPORTED_DATASETS)


class TestPathHelpers:
    """Tests for path helper functions."""

    def test_get_dataset_path(self, tmp_path: Path) -> None:
        """Test getting dataset path."""
        path: Path = get_dataset_path(tmp_path, "scifact")

        assert path == tmp_path / "raw" / "scifact"


class TestDatasetDownloadCheck:
    """Tests for checking if dataset is downloaded."""

    def test_is_dataset_downloaded_not_exists(self, tmp_path: Path) -> None:
        """Test returns False when directory doesn't exist."""
        assert not is_dataset_downloaded(tmp_path, "scifact")

    def test_is_dataset_downloaded_missing_files(self, tmp_path: Path) -> None:
        """Test returns False when required files are missing."""
        dataset_path: Path = tmp_path / "raw" / "scifact"
        dataset_path.mkdir(parents=True)

        # Create only corpus.jsonl
        (dataset_path / "corpus.jsonl").touch()

        assert not is_dataset_downloaded(tmp_path, "scifact")

    def test_is_dataset_downloaded_complete(self, tmp_path: Path) -> None:
        """Test returns True when all required files exist."""
        dataset_path: Path = tmp_path / "raw" / "scifact"
        dataset_path.mkdir(parents=True)

        # Create all required files
        (dataset_path / "corpus.jsonl").touch()
        (dataset_path / "queries.jsonl").touch()
        (dataset_path / "metadata.json").touch()

        # Create qrels directory with a file
        qrels_dir: Path = dataset_path / "qrels"
        qrels_dir.mkdir()
        (qrels_dir / "test.tsv").touch()

        assert is_dataset_downloaded(tmp_path, "scifact")


class TestLoadCorpus:
    """Tests for loading corpus."""

    def test_load_corpus(self, tmp_path: Path) -> None:
        """Test loading corpus from JSONL."""
        dataset_path: Path = tmp_path / "raw" / "scifact"
        dataset_path.mkdir(parents=True)

        # Create corpus file
        corpus_data: list[dict] = [
            {"_id": "doc1", "title": "Title 1", "text": "Text 1"},
            {"_id": "doc2", "title": "Title 2", "text": "Text 2", "extra": "data"},
        ]

        with open(dataset_path / "corpus.jsonl", "w") as f:
            for doc in corpus_data:
                f.write(json.dumps(doc) + "\n")

        corpus: dict[str, Document] = load_corpus(dataset_path)

        assert len(corpus) == 2
        assert corpus["doc1"].doc_id == "doc1"
        assert corpus["doc1"].title == "Title 1"
        assert corpus["doc1"].text == "Text 1"
        assert corpus["doc2"].metadata == {"extra": "data"}


class TestLoadQueries:
    """Tests for loading queries."""

    def test_load_queries(self, tmp_path: Path) -> None:
        """Test loading queries from JSONL."""
        dataset_path: Path = tmp_path / "raw" / "scifact"
        dataset_path.mkdir(parents=True)

        # Create queries file
        queries_data: list[dict] = [
            {"_id": "q1", "text": "Query 1"},
            {"_id": "q2", "text": "Query 2", "category": "science"},
        ]

        with open(dataset_path / "queries.jsonl", "w") as f:
            for query in queries_data:
                f.write(json.dumps(query) + "\n")

        queries: dict[str, Query] = load_queries(dataset_path)

        assert len(queries) == 2
        assert queries["q1"].query_id == "q1"
        assert queries["q1"].text == "Query 1"
        assert queries["q2"].metadata == {"category": "science"}


class TestLoadQrels:
    """Tests for loading qrels."""

    def test_load_qrels(self, tmp_path: Path) -> None:
        """Test loading qrels from TSV."""
        dataset_path: Path = tmp_path / "raw" / "scifact"
        qrels_dir: Path = dataset_path / "qrels"
        qrels_dir.mkdir(parents=True)

        # Create qrels file
        qrels_content: str = """query_id\tdoc_id\trelevance
q1\tdoc1\t1
q1\tdoc2\t2
q2\tdoc3\t1
"""
        with open(qrels_dir / "test.tsv", "w") as f:
            f.write(qrels_content)

        qrels: dict[str, dict[str, int]] = load_qrels(dataset_path, "test")

        assert len(qrels) == 2
        assert qrels["q1"]["doc1"] == 1
        assert qrels["q1"]["doc2"] == 2
        assert qrels["q2"]["doc3"] == 1

    def test_load_qrels_missing_file(self, tmp_path: Path) -> None:
        """Test loading qrels when file doesn't exist."""
        dataset_path: Path = tmp_path / "raw" / "scifact"
        dataset_path.mkdir(parents=True)

        qrels: dict[str, dict[str, int]] = load_qrels(dataset_path, "test")

        assert qrels == {}


class TestBeirDataset:
    """Tests for BeirDataset class."""

    def test_beir_dataset_methods(self) -> None:
        """Test BeirDataset helper methods."""
        from src.data.models import DatasetInfo

        corpus: dict[str, Document] = {
            "doc1": Document(doc_id="doc1", text="Text 1"),
            "doc2": Document(doc_id="doc2", text="Text 2"),
        }
        queries: dict[str, Query] = {
            "q1": Query(query_id="q1", text="Query 1"),
        }
        qrels: dict[str, dict[str, int]] = {
            "q1": {"doc1": 1, "doc2": 2},
        }
        info = DatasetInfo(
            name="test",
            num_documents=2,
            num_queries=1,
            num_qrels=2,
        )

        dataset = BeirDataset(
            name="test",
            corpus=corpus,
            queries=queries,
            qrels=qrels,
            info=info,
        )

        assert dataset.get_query_ids() == ["q1"]
        assert set(dataset.get_doc_ids()) == {"doc1", "doc2"}
        assert dataset.get_relevant_docs("q1") == {"doc1": 1, "doc2": 2}
        assert dataset.get_relevant_docs("q_nonexistent") == {}


class TestGetQrelsAsList:
    """Tests for qrels conversion."""

    def test_get_qrels_as_list(self) -> None:
        """Test converting nested qrels dict to list of Qrel objects."""
        qrels: dict[str, dict[str, int]] = {
            "q1": {"doc1": 1, "doc2": 2},
            "q2": {"doc3": 1},
        }

        qrels_list: list[Qrel] = get_qrels_as_list(qrels)

        assert len(qrels_list) == 3

        # Check all entries are present
        entries: set[tuple[str, str, int]] = {
            (q.query_id, q.doc_id, q.relevance) for q in qrels_list
        }
        assert ("q1", "doc1", 1) in entries
        assert ("q1", "doc2", 2) in entries
        assert ("q2", "doc3", 1) in entries


class TestLoadDataset:
    """Tests for loading full dataset."""

    def test_load_dataset_not_found_error(self, tmp_path: Path) -> None:
        """Test error when dataset not found and download disabled."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_dataset("scifact", tmp_path, download_if_missing=False)

    def test_load_dataset_with_existing_data(self, tmp_path: Path) -> None:
        """Test loading dataset from existing files."""
        dataset_path: Path = tmp_path / "raw" / "scifact"
        dataset_path.mkdir(parents=True)

        # Create corpus
        with open(dataset_path / "corpus.jsonl", "w") as f:
            f.write(json.dumps({"_id": "doc1", "title": "T", "text": "Text"}) + "\n")

        # Create queries
        with open(dataset_path / "queries.jsonl", "w") as f:
            f.write(json.dumps({"_id": "q1", "text": "Query"}) + "\n")

        # Create metadata
        with open(dataset_path / "metadata.json", "w") as f:
            json.dump({
                "name": "scifact",
                "num_documents": 1,
                "num_queries": 1,
                "num_qrels": 1,
                "download_timestamp": "2024-01-01T00:00:00",
            }, f)

        # Create qrels
        qrels_dir: Path = dataset_path / "qrels"
        qrels_dir.mkdir()
        with open(qrels_dir / "test.tsv", "w") as f:
            f.write("query_id\tdoc_id\trelevance\nq1\tdoc1\t1\n")

        dataset: BeirDataset = load_dataset(
            "scifact",
            tmp_path,
            download_if_missing=False,
        )

        assert dataset.name == "scifact"
        assert len(dataset.corpus) == 1
        assert len(dataset.queries) == 1
        assert "q1" in dataset.qrels
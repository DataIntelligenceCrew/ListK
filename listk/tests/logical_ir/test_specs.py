"""
Tests for solicedb.logical_ir.specs module.

Covers:
- SemanticSpec class and all its methods
- ALL_COLUMNS_MARKER constant
"""

import pytest

from solicedb.logical_ir.specs import ALL_COLUMNS_MARKER, SemanticSpec


class TestSemanticSpec:
    """Tests for the SemanticSpec class."""

    def test_parse_single_column(self) -> None:
        """Test parsing template with single column reference."""
        spec = SemanticSpec.parse("Is {title} a classic film?")
        assert spec.template == "Is {title} a classic film?"
        assert spec.input_columns == frozenset({"title"})
        assert spec.all_columns is False

    def test_parse_multiple_columns(self) -> None:
        """Test parsing template with multiple column references."""
        spec = SemanticSpec.parse("Is {title} from {year} a good movie?")
        assert spec.input_columns == frozenset({"title", "year"})
        assert spec.all_columns is False

    def test_parse_duplicate_column_references(self) -> None:
        """Test parsing template with duplicate column references."""
        spec = SemanticSpec.parse("{title} is {title}")
        assert spec.input_columns == frozenset({"title"})

    def test_parse_no_columns(self) -> None:
        """Test parsing template with no column references."""
        spec = SemanticSpec.parse("Classify this item")
        assert spec.input_columns == frozenset()
        assert spec.all_columns is False

    def test_parse_all_columns_marker(self) -> None:
        """Test parsing template with ALL_COLS marker."""
        spec = SemanticSpec.parse("Summarize this record: {ALL_COLS}")
        assert spec.all_columns is True
        assert ALL_COLUMNS_MARKER not in spec.input_columns

    def test_parse_mixed_columns_and_all_cols(self) -> None:
        """Test parsing with both specific columns and ALL_COLS."""
        spec = SemanticSpec.parse("Title: {title}, Details: {ALL_COLS}")
        assert spec.all_columns is True
        assert spec.input_columns == frozenset({"title"})

    def test_create_with_explicit_columns(self) -> None:
        """Test create() with explicit column dependencies."""
        spec = SemanticSpec.create(
            "Classify the genre",
            input_columns=["title", "synopsis"],
            all_columns=False,
        )
        assert spec.template == "Classify the genre"
        assert spec.input_columns == frozenset({"title", "synopsis"})

    def test_create_with_frozenset(self) -> None:
        """Test create() accepts frozenset."""
        spec = SemanticSpec.create(
            "Test",
            input_columns=frozenset({"a", "b"}),
        )
        assert spec.input_columns == frozenset({"a", "b"})

    def test_create_with_set(self) -> None:
        """Test create() accepts set."""
        spec = SemanticSpec.create(
            "Test",
            input_columns={"a", "b"},
        )
        assert spec.input_columns == frozenset({"a", "b"})

    def test_create_with_none_parses_template(self) -> None:
        """Test create() parses template when input_columns is None."""
        spec = SemanticSpec.create("Is {title} good?", input_columns=None)
        assert spec.input_columns == frozenset({"title"})

    def test_format_single_column(self) -> None:
        """Test formatting template with single column."""
        spec = SemanticSpec.parse("Is {title} a classic?")
        result = spec.format({"title": "The Matrix"})
        assert result == "Is The Matrix a classic?"

    def test_format_multiple_columns(self) -> None:
        """Test formatting template with multiple columns."""
        spec = SemanticSpec.parse("Is {title} from {year} good?")
        result = spec.format({"title": "The Matrix", "year": "1999"})
        assert result == "Is The Matrix from 1999 good?"

    def test_format_all_columns(self) -> None:
        """Test formatting template with ALL_COLS marker."""
        spec = SemanticSpec.parse("Record: {ALL_COLS}")
        result = spec.format({"title": "The Matrix", "year": "1999"})
        assert "title: The Matrix" in result
        assert "year: 1999" in result

    def test_format_missing_column_not_replaced(self) -> None:
        """Test that missing columns in row are not replaced."""
        spec = SemanticSpec.parse("Is {title} from {year} good?")
        result = spec.format({"title": "The Matrix"})
        assert "{year}" in result
        assert "The Matrix" in result

    def test_get_required_columns_explicit(self) -> None:
        """Test get_required_columns with explicit column references."""
        spec = SemanticSpec.parse("Is {title} from {year} good?")
        required = spec.get_required_columns()
        assert required == frozenset({"title", "year"})

    def test_get_required_columns_all_columns(self) -> None:
        """Test get_required_columns when all_columns=True."""
        spec = SemanticSpec.parse("Summarize: {ALL_COLS}")
        available = ["id", "title", "year"]
        required = spec.get_required_columns(available_columns=available)
        assert required == frozenset({"id", "title", "year"})

    def test_get_required_columns_all_columns_no_available_raises(self) -> None:
        """Test get_required_columns raises when all_columns but no available."""
        spec = SemanticSpec.parse("Summarize: {ALL_COLS}")
        with pytest.raises(ValueError, match="available_columns must be provided"):
            spec.get_required_columns()

    def test_validate_columns_all_present(self) -> None:
        """Test validate_columns when all columns are present."""
        spec = SemanticSpec.parse("Is {title} from {year} good?")
        missing = spec.validate_columns({"title", "year", "rating"})
        assert missing == []

    def test_validate_columns_some_missing(self) -> None:
        """Test validate_columns when some columns are missing."""
        spec = SemanticSpec.parse("Is {title} from {year} good?")
        missing = spec.validate_columns({"title"})
        assert missing == ["year"]

    def test_validate_columns_all_missing(self) -> None:
        """Test validate_columns when all columns are missing."""
        spec = SemanticSpec.parse("Is {title} from {year} good?")
        missing = spec.validate_columns({"rating"})
        assert sorted(missing) == ["title", "year"]

    def test_validate_columns_all_cols_always_valid(self) -> None:
        """Test validate_columns with ALL_COLS is always valid."""
        spec = SemanticSpec.parse("Summarize: {ALL_COLS}")
        missing = spec.validate_columns(set())
        assert missing == []

    def test_validate_columns_returns_sorted(self) -> None:
        """Test validate_columns returns sorted list of missing columns."""
        spec = SemanticSpec.parse("{z} {a} {m}")
        missing = spec.validate_columns(set())
        assert missing == ["a", "m", "z"]

    def test_to_string_explicit_columns(self) -> None:
        """Test to_string representation with explicit columns."""
        spec = SemanticSpec.parse("Is {title} good?")
        result = spec.to_string()
        assert "SemanticSpec" in result
        assert "Is {title} good?" in result
        assert "title" in result

    def test_to_string_all_columns(self) -> None:
        """Test to_string representation with all_columns."""
        spec = SemanticSpec.parse("Summarize: {ALL_COLS}")
        result = spec.to_string()
        assert "all_columns=True" in result

    def test_str_method(self) -> None:
        """Test __str__ delegates to to_string."""
        spec = SemanticSpec.parse("Is {title} good?")
        assert str(spec) == spec.to_string()

    def test_immutability(self) -> None:
        """Test that SemanticSpec is immutable."""
        spec = SemanticSpec.parse("Is {title} good?")
        with pytest.raises(AttributeError):
            spec.template = "new template"  # type: ignore

    def test_complex_template(self) -> None:
        """Test parsing complex template with many columns."""
        template = (
            "Given {title} ({year}), directed by {director}, "
            "with rating {rating}, classify its genre."
        )
        spec = SemanticSpec.parse(template)
        assert spec.input_columns == frozenset({"title", "year", "director", "rating"})


class TestAllColumnsMarker:
    """Tests for the ALL_COLUMNS_MARKER constant."""

    def test_all_columns_marker_value(self) -> None:
        """Test ALL_COLUMNS_MARKER has expected value."""
        assert ALL_COLUMNS_MARKER == "ALL_COLS"

    def test_all_columns_marker_in_template(self) -> None:
        """Test ALL_COLUMNS_MARKER can be used in templates."""
        template = f"Process: {{{ALL_COLUMNS_MARKER}}}"
        spec = SemanticSpec.parse(template)
        assert spec.all_columns is True